#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <array>
#include <vector>

#include "../third_party/nnue_pytorch/nnue_training_data_formats.h"

using binpack::CompressedTrainingDataEntryParallelReader;
using binpack::CompressedTrainingDataFile;
using binpack::PackedMoveScoreListReader;
using binpack::PackedTrainingDataEntry;
using binpack::CompressedTrainingDataEntryWriter;
using binpack::TrainingDataEntry;
using binpack::unpackEntry;

extern "C" {

struct ThrawnBatchView {
    std::int32_t size;
    std::int32_t max_active_features;
    std::int32_t* white_indices;
    std::int32_t* black_indices;
    float* stm;
    float* score;
    float* result_wdl;
};

struct ThrawnInspectStats {
    std::uint64_t entries_seen;
    std::uint64_t entries_skipped;
    std::uint64_t entries_read;
    std::uint64_t root_entries;
    std::uint64_t continuation_entries;
    std::uint64_t chunk_count;
    std::uint64_t chunk_payload_bytes;
    std::uint32_t min_chunk_payload_bytes;
    std::uint32_t max_chunk_payload_bytes;
    std::uint64_t white_to_move;
    std::uint64_t black_to_move;
    std::uint64_t wins;
    std::uint64_t draws;
    std::uint64_t losses;
    double min_score;
    double max_score;
    std::uint16_t min_ply;
    std::uint16_t max_ply;
    double mean_score;
    double score_std;
    double mean_raw_score;
    double raw_score_std;
    double mean_abs_score;
    double abs_score_std;
    double mean_ply;
    double ply_std;
    double mean_piece_count;
    double piece_count_std;
    double mean_non_king_piece_count;
    double result_mean;
    double result_wdl_mean;
    double score_result_correlation;
    double score_p01;
    double score_p05;
    double score_p10;
    double score_p25;
    double score_p50;
    double score_p75;
    double score_p90;
    double score_p95;
    double score_p99;
    double score_p999;
    double abs_score_p50;
    double abs_score_p75;
    double abs_score_p90;
    double abs_score_p95;
    double abs_score_p99;
    double abs_score_p999;
    double ply_p05;
    double ply_p25;
    double ply_p50;
    double ply_p75;
    double ply_p95;
    double ply_p99;
    double piece_count_p05;
    double piece_count_p25;
    double piece_count_p50;
    double piece_count_p75;
    double piece_count_p95;
    std::uint64_t abs_score_ge_1000;
    std::uint64_t abs_score_ge_2000;
    std::uint64_t abs_score_ge_4000;
    std::uint64_t abs_score_ge_8000;
    std::uint64_t abs_score_ge_16000;
    std::uint64_t score_bucket_counts[21];
    std::uint64_t abs_score_bucket_counts[11];
    std::uint64_t ply_bucket_counts[8];
    std::uint64_t piece_count_bucket_counts[4];
    std::uint64_t phase_counts[4];
    double phase_mean_score[4];
    double phase_mean_abs_score[4];
    double phase_result_mean[4];
    std::uint64_t result_score_agree;
    std::uint64_t result_score_disagree;
    std::uint64_t decisive_result_near_zero_score;
    std::uint64_t draw_high_abs_score;
    double mean_score_win;
    double mean_score_draw;
    double mean_score_loss;
    double mean_abs_score_win;
    double mean_abs_score_draw;
    double mean_abs_score_loss;
    std::uint64_t capture_positions;
    std::uint64_t in_check_positions;
    std::uint64_t move_type_counts[4];
    std::uint64_t piece_type_counts[6];
    std::uint64_t white_piece_counts[6];
    std::uint64_t black_piece_counts[6];
};

void* thrawn_binpack_open_many(
    const char* const* paths,
    std::int32_t num_paths,
    std::int32_t num_threads,
    std::int32_t cyclic,
    std::int32_t skip_wdl_score_mismatch,
    std::int32_t skip_tactical_positions,
    std::int32_t random_fen_skipping,
    std::int32_t early_fen_skipping,
    std::int32_t soft_early_fen_skipping,
    std::int32_t simple_eval_skipping,
    double pc_y0,
    double pc_y1,
    double pc_y2,
    double pc_y3,
    double pc_y4,
    double ply_x1,
    double ply_y1,
    double ply_x2,
    double ply_y2,
    double ply_x3,
    double ply_y3,
    double ply_x4,
    double ply_y4,
    std::int32_t split_role,
    double validation_split_fraction
);
void thrawn_binpack_close(void* handle);
ThrawnBatchView* thrawn_binpack_next_batch(void* handle, std::int32_t batch_size);
void thrawn_batch_free(ThrawnBatchView* batch);
std::int32_t thrawn_inspect_binpack(const char* path, ThrawnInspectStats* out_stats);
std::int32_t thrawn_inspect_binpack_with_options(
    const char* path,
    ThrawnInspectStats* out_stats,
    std::int32_t skip_wdl_score_mismatch,
    std::uint64_t sample_entries
);
std::int32_t thrawn_write_fixture_binpack(const char* path);
const char* thrawn_last_error();

}

namespace {

thread_local std::string g_last_error;

constexpr std::int32_t kHalfKaV2HmPieceSquareFeatures = 11 * 64;
constexpr std::int32_t kHalfKaV2HmMaxActiveFeatures = 32;
constexpr double kStockfishInternalToCp = 100.0 / 208.0;
constexpr std::array<std::int32_t, 64> kHalfKaV2HmKingBuckets{
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
    8, 9, 10, 11, 11, 10, 9, 8,
    4, 5, 6, 7, 7, 6, 5, 4,
    0, 1, 2, 3, 3, 2, 1, 0,
};
constexpr std::array<chess::PieceType, 6> kPieceTypes{
    chess::PieceType::Pawn,
    chess::PieceType::Knight,
    chess::PieceType::Bishop,
    chess::PieceType::Rook,
    chess::PieceType::Queen,
    chess::PieceType::King,
};

struct BinpackFilterOptions {
    bool skip_wdl_score_mismatch = false;
    bool skip_tactical_positions = false;
    std::int32_t random_fen_skipping = 0;
    std::int32_t early_fen_skipping = -1;
    std::int32_t soft_early_fen_skipping = 0;
    std::int32_t simple_eval_skipping = -1;
    double pc_y0 = 0.0;
    double pc_y1 = 0.4;
    double pc_y2 = 1.0;
    double pc_y3 = 1.0;
    double pc_y4 = 0.75;
    double ply_x1 = 0.0;
    double ply_y1 = 0.1;
    double ply_x2 = 6.0;
    double ply_y2 = 0.15;
    double ply_x3 = 10.0;
    double ply_y3 = 0.25;
    double ply_x4 = 18.0;
    double ply_y4 = 0.75;
    std::int32_t split_role = 0;
    double validation_split_fraction = 0.0;
    std::uint64_t sample_entries = 0;
};

[[nodiscard]] std::function<bool(const TrainingDataEntry&)> make_skip_predicate(BinpackFilterOptions options);

struct ReaderHandle {
    explicit ReaderHandle(
        std::vector<std::string> input_paths,
        std::int32_t num_threads,
        bool cyclic,
        BinpackFilterOptions filter_options
    ) :
        paths(std::move(input_paths)),
        reader(std::make_unique<CompressedTrainingDataEntryParallelReader>(
            std::max(1, num_threads),
            paths,
            std::ios_base::binary,
            cyclic,
            make_skip_predicate(filter_options)
        ))
    {}

    std::vector<std::string> paths;
    std::unique_ptr<CompressedTrainingDataEntryParallelReader> reader;
};

[[nodiscard]] std::int32_t halfka_v2_hm_orient(
    chess::Color perspective,
    chess::Square sq,
    chess::Square king_sq
) {
    std::int32_t result = static_cast<std::int32_t>(sq);
    if (static_cast<int>(king_sq.file()) < 4) {
        result ^= 7;
    }
    if (perspective == chess::Color::Black) {
        result ^= 56;
    }
    return result;
}

[[nodiscard]] std::int32_t halfka_v2_hm_piece_square_offset(chess::Color perspective, chess::Piece piece) {
    std::int32_t piece_bucket = 0;
    switch (piece.type()) {
    case chess::PieceType::Pawn:
        piece_bucket = 0;
        break;
    case chess::PieceType::Knight:
        piece_bucket = 2;
        break;
    case chess::PieceType::Bishop:
        piece_bucket = 4;
        break;
    case chess::PieceType::Rook:
        piece_bucket = 6;
        break;
    case chess::PieceType::Queen:
        piece_bucket = 8;
        break;
    case chess::PieceType::King:
        return 10 * 64;
    default:
        throw std::runtime_error("unsupported piece type in HalfKAv2_hm feature extraction");
    }
    piece_bucket += static_cast<std::int32_t>(piece.color() != perspective);
    return piece_bucket * 64;
}

[[nodiscard]] std::int32_t halfka_v2_hm_feature_index(
    chess::Color perspective,
    chess::Square king_sq,
    chess::Square sq,
    chess::Piece piece
) {
    const std::int32_t flip = perspective == chess::Color::Black ? 56 : 0;
    const auto king_bucket = kHalfKaV2HmKingBuckets[static_cast<std::size_t>(static_cast<int>(king_sq) ^ flip)];
    return halfka_v2_hm_orient(perspective, sq, king_sq) +
           halfka_v2_hm_piece_square_offset(perspective, piece) +
           king_bucket * kHalfKaV2HmPieceSquareFeatures;
}

[[nodiscard]] std::int32_t fill_halfka_v2_hm_feature_list(
    const chess::Position& pos,
    chess::Color perspective,
    std::int32_t* out_features
) {
    const auto king_sq = pos.kingSquare(perspective);
    std::int32_t count = 0;
    for (chess::Square sq : pos.piecesBB()) {
        const auto piece = pos.pieceAt(sq);
        if (piece == chess::Piece::none()) {
            continue;
        }
        if (count >= kHalfKaV2HmMaxActiveFeatures) {
            break;
        }
        out_features[count++] = halfka_v2_hm_feature_index(perspective, king_sq, sq, piece);
    }
    return count;
}

[[nodiscard]] float result_to_wdl(std::int16_t result) {
    if (result > 0) {
        return 1.0f;
    }
    if (result < 0) {
        return 0.0f;
    }
    return 0.5f;
}

[[nodiscard]] double score_to_cp(double score) {
    return score * kStockfishInternalToCp;
}

[[nodiscard]] std::uint64_t mix_u64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void hash_combine(std::uint64_t& seed, std::uint64_t value) {
    seed ^= mix_u64(value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

[[nodiscard]] double stable_unit_random(const TrainingDataEntry& entry, std::uint64_t salt = 0) {
    std::uint64_t seed = 0xcbf29ce484222325ULL ^ salt;
    for (chess::Square sq : entry.pos.piecesBB()) {
        const auto piece = entry.pos.pieceAt(sq);
        hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(sq)));
        hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(piece)));
    }
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(entry.pos.sideToMove())));
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<std::uint16_t>(entry.score)));
    hash_combine(seed, static_cast<std::uint64_t>(entry.ply));
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<std::uint16_t>(entry.result)));
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(entry.move.from)));
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(entry.move.to)));
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(entry.move.type)));
    hash_combine(seed, static_cast<std::uint64_t>(static_cast<int>(entry.move.promotedPiece)));
    return std::ldexp(static_cast<double>(mix_u64(seed) >> 11), -53);
}

[[nodiscard]] std::mt19937_64& thread_rng() {
    thread_local std::mt19937_64 rng(std::random_device{}());
    return rng;
}

[[nodiscard]] double random_unit() {
    return std::ldexp(static_cast<double>(thread_rng()() >> 11), -53);
}

[[nodiscard]] std::function<bool(const TrainingDataEntry&)> make_skip_predicate(BinpackFilterOptions options) {
    const bool has_stockfish_filter =
        options.skip_tactical_positions ||
        options.skip_wdl_score_mismatch ||
        options.random_fen_skipping > 0 ||
        options.early_fen_skipping >= 0 ||
        options.soft_early_fen_skipping > 0 ||
        options.simple_eval_skipping > 0;
    const bool has_local_filter =
        (options.validation_split_fraction > 0.0 && options.split_role != 0);
    if (!has_stockfish_filter && !has_local_filter) {
        return nullptr;
    }

    std::uint64_t random_skip_threshold = 0;
    if (options.random_fen_skipping > 0) {
        const double skip_prob =
            static_cast<double>(options.random_fen_skipping) /
            static_cast<double>(options.random_fen_skipping + 1);
        random_skip_threshold = static_cast<std::uint64_t>(
            skip_prob * static_cast<double>(std::numeric_limits<std::uint64_t>::max())
        );
    }

    std::array<double, 33> target_pc_weights_lut{};
    double target_pc_weights_total = 0.0;

    auto desired_piece_count_weight = [&options](int pc) -> double {
        const double x = static_cast<double>(pc);
        const double y[5] = {options.pc_y0, options.pc_y1, options.pc_y2, options.pc_y3, options.pc_y4};

        if (x <= 0.0) {
            return y[0];
        }
        if (x >= 32.0) {
            return y[4];
        }

        int i = static_cast<int>(x / 8.0);
        if (i > 3) {
            i = 3;
        }

        const double x0 = static_cast<double>(i) * 8.0;
        const double t = (x - x0) / 8.0;

        auto get_slope = [&](int idx) {
            if (idx == 0) {
                return y[1] - y[0];
            }
            if (idx == 4) {
                return y[4] - y[3];
            }
            return (y[idx + 1] - y[idx - 1]) / 2.0;
        };

        const double m0 = get_slope(i);
        const double m1 = get_slope(i + 1);
        const double t2 = t * t;
        const double t3 = t2 * t;
        const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const double h10 = t3 - 2.0 * t2 + t;
        const double h01 = -2.0 * t3 + 3.0 * t2;
        const double h11 = t3 - t2;

        const double value = h00 * y[i] + h10 * m0 + h01 * y[i + 1] + h11 * m1;
        return std::max(0.0, value);
    };

    for (int i = 0; i < 33; ++i) {
        target_pc_weights_lut[static_cast<std::size_t>(i)] = desired_piece_count_weight(i);
        target_pc_weights_total += target_pc_weights_lut[static_cast<std::size_t>(i)];
    }
    if (target_pc_weights_total <= 0.0) {
        target_pc_weights_lut.fill(1.0);
        target_pc_weights_total = static_cast<double>(target_pc_weights_lut.size());
    }

    std::vector<double> early_ply_accept_prob;
    if (options.soft_early_fen_skipping > 0) {
        const auto lut_size = static_cast<std::size_t>(options.soft_early_fen_skipping) + 1;
        early_ply_accept_prob.resize(lut_size);

        auto interpolate_ply = [&options](double ply) -> double {
            struct Point {
                double x;
                double y;
            };
            const Point points[5] = {
                {options.ply_x1, options.ply_y1},
                {options.ply_x2, options.ply_y2},
                {options.ply_x3, options.ply_y3},
                {options.ply_x4, options.ply_y4},
                {static_cast<double>(options.soft_early_fen_skipping), 1.0},
            };

            if (ply <= points[0].x) {
                return points[0].y;
            }
            if (ply >= points[4].x) {
                return points[4].y;
            }
            for (int i = 0; i < 4; ++i) {
                if (ply >= points[i].x && ply <= points[i + 1].x) {
                    if (points[i + 1].x == points[i].x) {
                        return points[i].y;
                    }
                    const double t = (ply - points[i].x) / (points[i + 1].x - points[i].x);
                    return points[i].y + t * (points[i + 1].y - points[i].y);
                }
            }
            return 1.0;
        };

        for (std::size_t i = 0; i < early_ply_accept_prob.size(); ++i) {
            early_ply_accept_prob[i] = std::clamp(interpolate_ply(static_cast<double>(i)), 0.0, 1.0);
        }
    }

    return [
        options,
        has_stockfish_filter,
        random_skip_threshold,
        target_pc_weights_lut,
        target_pc_weights_total,
        early_ply_accept_prob = std::move(early_ply_accept_prob)
    ](const TrainingDataEntry& entry) {
        if (has_stockfish_filter) {
            static constexpr int kValueNone = 32002;
            static thread_local int last_ply = -1;
            static thread_local int last_score = kValueNone;

            const bool skip_placeholder_zero =
                entry.ply > last_ply &&
                last_score != kValueNone &&
                std::abs(last_score) > 100 &&
                entry.result != 0 &&
                entry.score == 0;

            last_ply = entry.ply;
            if (entry.score == kValueNone || skip_placeholder_zero) {
                return true;
            }
            last_score = entry.score;

            if (entry.ply <= options.early_fen_skipping) {
                return true;
            }
        }

        if (options.validation_split_fraction > 0.0 && options.split_role != 0) {
            const bool is_validation_entry =
                stable_unit_random(entry, 0x9e3779b97f4a7c15ULL) < options.validation_split_fraction;
            if (options.split_role == 1 && is_validation_entry) {
                return true;
            }
            if (options.split_role == 2 && !is_validation_entry) {
                return true;
            }
        }

        if (!has_stockfish_filter) {
            return false;
        }

        auto& prng = thread_rng();

        if (options.split_role != 2 && options.random_fen_skipping > 0 && prng() < random_skip_threshold) {
            return true;
        }
        if (options.skip_tactical_positions && (entry.isCapturingMove() || entry.isInCheck())) {
            return true;
        }
        if (options.skip_wdl_score_mismatch) {
            const double keep_probability = std::clamp(entry.score_result_prob(), 0.0, 1.0);
            const double draw = options.split_role == 2
                ? stable_unit_random(entry, 0xd1b54a32d192ed03ULL)
                : random_unit();
            if (draw >= keep_probability) {
                return true;
            }
        }
        if (options.simple_eval_skipping > 0 && std::abs(entry.pos.simple_eval()) < options.simple_eval_skipping) {
            return true;
        }
        if (options.soft_early_fen_skipping > 0 && entry.ply < options.soft_early_fen_skipping) {
            const double accept_probability = early_ply_accept_prob[entry.ply];
            const double draw = options.split_role == 2
                ? stable_unit_random(entry, 0xa24baed4963ee407ULL)
                : random_unit();
            if (draw >= accept_probability) {
                return true;
            }
        }

        const int piece_count = static_cast<int>(entry.pos.piecesBB().count());
        if (piece_count < 0 || piece_count > 32) {
            return true;
        }

        static thread_local double alpha = 1.0;
        static thread_local double pc_history_all[33] = {0.0};
        static thread_local double pc_history_passed[33] = {0.0};
        static thread_local double pc_history_all_total = 0.0;
        static thread_local double pc_history_passed_total = 0.0;
        static thread_local std::uint64_t step_count = 0;

        constexpr double kMaxPieceCountSkipRate = 0.975;

        pc_history_all[piece_count] += 1.0;
        pc_history_all_total += 1.0;
        ++step_count;

        const bool should_update =
            step_count == 100 ||
            step_count == 500 ||
            step_count == 1000 ||
            step_count == 2500 ||
            step_count == 5000 ||
            (step_count > 5000 && step_count % 10000 == 0);

        if (should_update) {
            double min_ratio = std::numeric_limits<double>::infinity();
            bool found_valid = false;
            for (int i = 0; i < 33; ++i) {
                const auto index = static_cast<std::size_t>(i);
                if (target_pc_weights_lut[index] > 0.0 && pc_history_all[i] > 0.0) {
                    const double current_ratio =
                        (pc_history_all_total * target_pc_weights_lut[index]) /
                        (target_pc_weights_total * pc_history_all[i]);
                    if (current_ratio < min_ratio) {
                        min_ratio = current_ratio;
                        found_valid = true;
                    }
                }
            }
            if (found_valid && min_ratio > 0.0) {
                alpha = (1.0 - kMaxPieceCountSkipRate) / min_ratio;
            }

            if (step_count >= 10000 && step_count % 10000 == 0) {
                for (double& count : pc_history_all) {
                    count *= 0.5;
                }
                pc_history_all_total *= 0.5;
            }
        }

        double accept_probability = 0.0;
        const auto pc_index = static_cast<std::size_t>(piece_count);
        if (target_pc_weights_lut[pc_index] > 0.0 && pc_history_all[piece_count] > 0.0) {
            const double current_ratio =
                (pc_history_all_total * target_pc_weights_lut[pc_index]) /
                (target_pc_weights_total * pc_history_all[piece_count]);
            accept_probability = alpha * current_ratio;
        }
        accept_probability = std::clamp(accept_probability, 0.0, 1.0);

        const double draw = options.split_role == 2
            ? stable_unit_random(entry, 0x9fb21c651e98df25ULL)
            : random_unit();
        if (draw >= accept_probability) {
            return true;
        }

        pc_history_passed[piece_count] += 1.0;
        pc_history_passed_total += 1.0;
        return false;
    };
}

[[nodiscard]] double population_std(double sum, double sum_sq, std::uint64_t count) {
    if (count == 0) {
        return 0.0;
    }
    const double mean = sum / static_cast<double>(count);
    const double variance = (sum_sq / static_cast<double>(count)) - (mean * mean);
    return std::sqrt(std::max(0.0, variance));
}

[[nodiscard]] int score_bucket_index(double score_cp) {
    if (score_cp < -8000.0) return 0;
    if (score_cp < -4000.0) return 1;
    if (score_cp < -2000.0) return 2;
    if (score_cp < -1000.0) return 3;
    if (score_cp < -600.0) return 4;
    if (score_cp < -400.0) return 5;
    if (score_cp < -200.0) return 6;
    if (score_cp < -100.0) return 7;
    if (score_cp < -50.0) return 8;
    if (score_cp < 0.0) return 9;
    if (score_cp == 0.0) return 10;
    if (score_cp <= 50.0) return 11;
    if (score_cp <= 100.0) return 12;
    if (score_cp <= 200.0) return 13;
    if (score_cp <= 400.0) return 14;
    if (score_cp <= 600.0) return 15;
    if (score_cp <= 1000.0) return 16;
    if (score_cp <= 2000.0) return 17;
    if (score_cp <= 4000.0) return 18;
    if (score_cp <= 8000.0) return 19;
    return 20;
}

[[nodiscard]] int abs_score_bucket_index(double abs_score_cp) {
    if (abs_score_cp < 50.0) return 0;
    if (abs_score_cp < 100.0) return 1;
    if (abs_score_cp < 200.0) return 2;
    if (abs_score_cp < 400.0) return 3;
    if (abs_score_cp < 600.0) return 4;
    if (abs_score_cp < 1000.0) return 5;
    if (abs_score_cp < 2000.0) return 6;
    if (abs_score_cp < 4000.0) return 7;
    if (abs_score_cp < 8000.0) return 8;
    if (abs_score_cp < 16000.0) return 9;
    return 10;
}

[[nodiscard]] int ply_bucket_index(std::uint16_t ply) {
    if (ply < 20) return 0;
    if (ply < 40) return 1;
    if (ply < 60) return 2;
    if (ply < 80) return 3;
    if (ply < 100) return 4;
    if (ply < 150) return 5;
    if (ply < 200) return 6;
    return 7;
}

[[nodiscard]] int piece_count_bucket_index(int piece_count) {
    if (piece_count <= 7) return 0;
    if (piece_count <= 15) return 1;
    if (piece_count <= 23) return 2;
    return 3;
}

[[nodiscard]] int phase_bucket_index(int piece_count) {
    if (piece_count <= 10) return 0;
    if (piece_count <= 16) return 1;
    if (piece_count <= 24) return 2;
    return 3;
}

template <std::size_t N>
[[nodiscard]] double percentile_from_histogram(
    const std::array<std::uint64_t, N>& histogram,
    std::uint64_t total,
    double q,
    int offset = 0
) {
    if (total == 0) {
        return 0.0;
    }
    const double clamped = std::clamp(q, 0.0, 1.0);
    const std::uint64_t target = static_cast<std::uint64_t>(std::floor(clamped * static_cast<double>(total - 1)));
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < histogram.size(); ++i) {
        cumulative += histogram[i];
        if (cumulative > target) {
            return static_cast<double>(static_cast<int>(i) + offset);
        }
    }
    return static_cast<double>(static_cast<int>(histogram.size() - 1) + offset);
}

template <typename T>
[[nodiscard]] T* alloc_array(std::size_t size) {
    return size == 0 ? nullptr : new T[size];
}

void clear_error() {
    g_last_error.clear();
}

void store_error(const std::exception& ex) {
    g_last_error = ex.what();
}

void store_error(const char* message) {
    g_last_error = message;
}

TrainingDataEntry make_entry(
    const char* fen,
    const char* uci,
    std::int16_t score,
    std::uint16_t ply,
    std::int16_t result
) {
    TrainingDataEntry entry;
    entry.pos = chess::Position::fromFen(fen);
    entry.move = chess::uci::uciToMove(entry.pos, uci);
    entry.score = score;
    entry.ply = ply;
    entry.result = result;
    return entry;
}

}  // namespace

extern "C" void* thrawn_binpack_open_many(
    const char* const* paths,
    std::int32_t num_paths,
    std::int32_t num_threads,
    std::int32_t cyclic,
    std::int32_t skip_wdl_score_mismatch,
    std::int32_t skip_tactical_positions,
    std::int32_t random_fen_skipping,
    std::int32_t early_fen_skipping,
    std::int32_t soft_early_fen_skipping,
    std::int32_t simple_eval_skipping,
    double pc_y0,
    double pc_y1,
    double pc_y2,
    double pc_y3,
    double pc_y4,
    double ply_x1,
    double ply_y1,
    double ply_x2,
    double ply_y2,
    double ply_x3,
    double ply_y3,
    double ply_x4,
    double ply_y4,
    std::int32_t split_role,
    double validation_split_fraction
) {
    clear_error();
    try {
        if (num_paths <= 0) {
            throw std::runtime_error("thrawn_binpack_open_many requires at least one dataset path");
        }
        if (split_role < 0 || split_role > 2) {
            throw std::runtime_error("split_role must be 0, 1, or 2");
        }
        if (validation_split_fraction < 0.0 || validation_split_fraction >= 1.0) {
            throw std::runtime_error("validation_split_fraction must be >= 0 and < 1");
        }
        if (random_fen_skipping < 0) {
            throw std::runtime_error("random_fen_skipping must be >= 0");
        }
        if (early_fen_skipping < -1) {
            throw std::runtime_error("early_fen_skipping must be >= -1");
        }
        if (simple_eval_skipping < -1) {
            throw std::runtime_error("simple_eval_skipping must be >= -1");
        }
        std::vector<std::string> owned_paths;
        owned_paths.reserve(static_cast<std::size_t>(num_paths));
        for (std::int32_t i = 0; i < num_paths; ++i) {
            if (paths[i] == nullptr) {
                throw std::runtime_error("dataset path must not be null");
            }
            owned_paths.emplace_back(paths[i]);
        }

        BinpackFilterOptions filter_options;
        filter_options.skip_wdl_score_mismatch = skip_wdl_score_mismatch != 0;
        filter_options.skip_tactical_positions = skip_tactical_positions != 0;
        filter_options.random_fen_skipping = random_fen_skipping;
        filter_options.early_fen_skipping = early_fen_skipping;
        filter_options.soft_early_fen_skipping = soft_early_fen_skipping;
        filter_options.simple_eval_skipping = simple_eval_skipping;
        filter_options.pc_y0 = pc_y0;
        filter_options.pc_y1 = pc_y1;
        filter_options.pc_y2 = pc_y2;
        filter_options.pc_y3 = pc_y3;
        filter_options.pc_y4 = pc_y4;
        filter_options.ply_x1 = ply_x1;
        filter_options.ply_y1 = ply_y1;
        filter_options.ply_x2 = ply_x2;
        filter_options.ply_y2 = ply_y2;
        filter_options.ply_x3 = ply_x3;
        filter_options.ply_y3 = ply_y3;
        filter_options.ply_x4 = ply_x4;
        filter_options.ply_y4 = ply_y4;
        filter_options.split_role = split_role;
        filter_options.validation_split_fraction = validation_split_fraction;

        auto* handle = new ReaderHandle(
            std::move(owned_paths),
            num_threads,
            cyclic != 0,
            filter_options
        );
        return handle;
    } catch (const std::exception& ex) {
        store_error(ex);
        return nullptr;
    }
}

extern "C" void thrawn_binpack_close(void* handle) {
    delete static_cast<ReaderHandle*>(handle);
}

extern "C" ThrawnBatchView* thrawn_binpack_next_batch(void* handle, std::int32_t batch_size) {
    clear_error();
    try {
        if (handle == nullptr) {
            throw std::runtime_error("reader handle is null");
        }
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be positive");
        }

        auto* reader_handle = static_cast<ReaderHandle*>(handle);
        std::vector<TrainingDataEntry> entries;
        entries.reserve(static_cast<std::size_t>(batch_size));
        const int filled = reader_handle->reader->fill(entries, static_cast<std::size_t>(batch_size));
        if (filled <= 0) {
            return nullptr;
        }

        auto* batch = new ThrawnBatchView{};
        const auto active_feature_count = kHalfKaV2HmMaxActiveFeatures;
        batch->size = filled;
        batch->max_active_features = active_feature_count;
        batch->white_indices = alloc_array<std::int32_t>(static_cast<std::size_t>(filled) * active_feature_count);
        batch->black_indices = alloc_array<std::int32_t>(static_cast<std::size_t>(filled) * active_feature_count);
        batch->stm = alloc_array<float>(filled);
        batch->score = alloc_array<float>(filled);
        batch->result_wdl = alloc_array<float>(filled);

        std::fill(
            batch->white_indices,
            batch->white_indices + static_cast<std::size_t>(filled) * active_feature_count,
            -1
        );
        std::fill(
            batch->black_indices,
            batch->black_indices + static_cast<std::size_t>(filled) * active_feature_count,
            -1
        );

        for (int i = 0; i < filled; ++i) {
            const auto& entry = entries[static_cast<std::size_t>(i)];
            const auto base = static_cast<std::size_t>(i) * active_feature_count;

            static_cast<void>(fill_halfka_v2_hm_feature_list(
                entry.pos,
                chess::Color::White,
                batch->white_indices + base
            ));
            static_cast<void>(fill_halfka_v2_hm_feature_list(
                entry.pos,
                chess::Color::Black,
                batch->black_indices + base
            ));
            batch->stm[i] = entry.pos.sideToMove() == chess::Color::White ? 1.0f : 0.0f;
            batch->score[i] = static_cast<float>(entry.score);
            batch->result_wdl[i] = result_to_wdl(entry.result);
        }

        return batch;
    } catch (const std::exception& ex) {
        store_error(ex);
        return nullptr;
    }
}

extern "C" void thrawn_batch_free(ThrawnBatchView* batch) {
    if (batch == nullptr) {
        return;
    }

    delete[] batch->white_indices;
    delete[] batch->black_indices;
    delete[] batch->stm;
    delete[] batch->score;
    delete[] batch->result_wdl;
    delete batch;
}

std::int32_t inspect_binpack_impl(
    const char* path,
    ThrawnInspectStats* out_stats,
    BinpackFilterOptions filter_options
) {
    clear_error();
    try {
        if (path == nullptr) {
            throw std::runtime_error("inspect path must not be null");
        }
        if (out_stats == nullptr) {
            throw std::runtime_error("inspect output struct must not be null");
        }

        CompressedTrainingDataFile input_file(path, std::ios_base::in);

        ThrawnInspectStats stats{};
        std::int16_t min_score_raw = std::numeric_limits<std::int16_t>::max();
        std::int16_t max_score_raw = std::numeric_limits<std::int16_t>::min();
        stats.min_ply = std::numeric_limits<std::uint16_t>::max();
        stats.max_ply = std::numeric_limits<std::uint16_t>::min();
        stats.min_chunk_payload_bytes = std::numeric_limits<std::uint32_t>::max();

        double total_raw_score = 0.0;
        double total_raw_score_sq = 0.0;
        double total_score = 0.0;
        double total_score_sq = 0.0;
        double total_abs_score = 0.0;
        double total_abs_score_sq = 0.0;
        double total_ply = 0.0;
        double total_ply_sq = 0.0;
        double total_piece_count = 0.0;
        double total_piece_count_sq = 0.0;
        double total_non_king_piece_count = 0.0;
        double total_result = 0.0;
        double total_result_sq = 0.0;
        double total_result_wdl = 0.0;
        double total_score_result = 0.0;
        auto score_histogram = std::make_unique<std::array<std::uint64_t, 65536>>();
        auto abs_score_histogram = std::make_unique<std::array<std::uint64_t, 32769>>();
        auto ply_histogram = std::make_unique<std::array<std::uint64_t, 65536>>();
        score_histogram->fill(0);
        abs_score_histogram->fill(0);
        ply_histogram->fill(0);
        std::array<std::uint64_t, 33> piece_count_histogram{};
        std::array<double, 4> phase_score_sums{};
        std::array<double, 4> phase_abs_score_sums{};
        std::array<double, 4> phase_result_sums{};
        std::array<std::uint64_t, 3> result_counts{};
        std::array<double, 3> result_score_sums{};
        std::array<double, 3> result_abs_score_sums{};
        auto skip_predicate = make_skip_predicate(filter_options);
        bool reached_sample_limit = false;

        auto sample_limit_reached = [&]() {
            return filter_options.sample_entries > 0 && stats.entries_seen >= filter_options.sample_entries;
        };

        auto record_entry = [&](const TrainingDataEntry& entry, bool is_continuation) {
            ++stats.entries_seen;
            if (skip_predicate && skip_predicate(entry)) {
                ++stats.entries_skipped;
                return;
            }

            ++stats.entries_read;
            if (is_continuation) {
                ++stats.continuation_entries;
            } else {
                ++stats.root_entries;
            }
            if (entry.pos.sideToMove() == chess::Color::White) {
                ++stats.white_to_move;
            } else {
                ++stats.black_to_move;
            }

            if (entry.result > 0) {
                ++stats.wins;
                ++result_counts[0];
            } else if (entry.result < 0) {
                ++stats.losses;
                ++result_counts[2];
            } else {
                ++stats.draws;
                ++result_counts[1];
            }

            min_score_raw = std::min(min_score_raw, entry.score);
            max_score_raw = std::max(max_score_raw, entry.score);
            stats.min_ply = std::min(stats.min_ply, entry.ply);
            stats.max_ply = std::max(stats.max_ply, entry.ply);

            const auto abs_score = static_cast<int>(std::abs(static_cast<int>(entry.score)));
            const double score_cp = score_to_cp(static_cast<double>(entry.score));
            const double abs_score_cp = score_to_cp(static_cast<double>(abs_score));
            const double ply = static_cast<double>(entry.ply);
            const double result_value = entry.result > 0 ? 1.0 : (entry.result < 0 ? -1.0 : 0.0);
            const double result_wdl_value = entry.result > 0 ? 1.0 : (entry.result < 0 ? 0.0 : 0.5);
            int piece_count = 0;
            for (std::size_t piece_index = 0; piece_index < kPieceTypes.size(); ++piece_index) {
                const auto piece_type = kPieceTypes[piece_index];
                const auto white_count = static_cast<std::uint64_t>(
                    entry.pos.piecesBB(chess::Piece(piece_type, chess::Color::White)).count()
                );
                const auto black_count = static_cast<std::uint64_t>(
                    entry.pos.piecesBB(chess::Piece(piece_type, chess::Color::Black)).count()
                );
                piece_count += static_cast<int>(white_count + black_count);
                stats.piece_type_counts[piece_index] += white_count + black_count;
                stats.white_piece_counts[piece_index] += white_count;
                stats.black_piece_counts[piece_index] += black_count;
            }
            const int non_king_piece_count = std::max(0, piece_count - 2);
            const int phase_bucket = phase_bucket_index(piece_count);
            const int result_bucket = entry.result > 0 ? 0 : (entry.result < 0 ? 2 : 1);
            const auto move_type = static_cast<int>(entry.move.type);

            total_raw_score += static_cast<double>(entry.score);
            total_raw_score_sq += static_cast<double>(entry.score) * static_cast<double>(entry.score);
            total_score += score_cp;
            total_score_sq += score_cp * score_cp;
            total_abs_score += abs_score_cp;
            total_abs_score_sq += abs_score_cp * abs_score_cp;
            total_ply += ply;
            total_ply_sq += ply * ply;
            total_piece_count += static_cast<double>(piece_count);
            total_piece_count_sq += static_cast<double>(piece_count * piece_count);
            total_non_king_piece_count += static_cast<double>(non_king_piece_count);
            total_result += result_value;
            total_result_sq += result_value * result_value;
            total_result_wdl += result_wdl_value;
            total_score_result += score_cp * result_value;
            (*score_histogram)[static_cast<std::uint16_t>(static_cast<int>(entry.score) + 32768)] += 1;
            (*abs_score_histogram)[std::min(abs_score, 32768)] += 1;
            (*ply_histogram)[entry.ply] += 1;
            piece_count_histogram[std::min(piece_count, 32)] += 1;
            stats.score_bucket_counts[score_bucket_index(score_cp)] += 1;
            stats.abs_score_bucket_counts[abs_score_bucket_index(abs_score_cp)] += 1;
            stats.ply_bucket_counts[ply_bucket_index(entry.ply)] += 1;
            stats.piece_count_bucket_counts[piece_count_bucket_index(piece_count)] += 1;
            stats.phase_counts[phase_bucket] += 1;
            phase_score_sums[phase_bucket] += score_cp;
            phase_abs_score_sums[phase_bucket] += abs_score_cp;
            phase_result_sums[phase_bucket] += result_value;
            result_score_sums[result_bucket] += score_cp;
            result_abs_score_sums[result_bucket] += abs_score_cp;
            stats.abs_score_ge_1000 += abs_score_cp >= 1000.0;
            stats.abs_score_ge_2000 += abs_score_cp >= 2000.0;
            stats.abs_score_ge_4000 += abs_score_cp >= 4000.0;
            stats.abs_score_ge_8000 += abs_score_cp >= 8000.0;
            stats.abs_score_ge_16000 += abs_score_cp >= 16000.0;

            if ((entry.result > 0 && score_cp > 0.0) || (entry.result < 0 && score_cp < 0.0)) {
                ++stats.result_score_agree;
            } else if ((entry.result > 0 && score_cp < 0.0) || (entry.result < 0 && score_cp > 0.0)) {
                ++stats.result_score_disagree;
            }
            if (entry.result != 0 && abs_score_cp < 50.0) {
                ++stats.decisive_result_near_zero_score;
            }
            if (entry.result == 0 && abs_score_cp >= 1000.0) {
                ++stats.draw_high_abs_score;
            }

            if (entry.isCapturingMove()) {
                ++stats.capture_positions;
            }
            if (entry.isInCheck()) {
                ++stats.in_check_positions;
            }
            if (move_type >= 0 && move_type < 4) {
                ++stats.move_type_counts[static_cast<std::size_t>(move_type)];
            }
        };

        while (input_file.hasNextChunk()) {
            if (sample_limit_reached()) {
                reached_sample_limit = true;
                break;
            }
            auto chunk = input_file.readNextChunk();
            const auto chunk_size = static_cast<std::uint32_t>(chunk.size());
            ++stats.chunk_count;
            stats.chunk_payload_bytes += chunk_size;
            stats.min_chunk_payload_bytes = std::min(stats.min_chunk_payload_bytes, chunk_size);
            stats.max_chunk_payload_bytes = std::max(stats.max_chunk_payload_bytes, chunk_size);

            std::size_t offset = 0;
            while (offset + sizeof(PackedTrainingDataEntry) + 2 <= chunk.size()) {
                if (sample_limit_reached()) {
                    reached_sample_limit = true;
                    break;
                }
                PackedTrainingDataEntry packed;
                std::memcpy(&packed, chunk.data() + offset, sizeof(PackedTrainingDataEntry));
                offset += sizeof(PackedTrainingDataEntry);

                const std::uint16_t num_plies = (chunk[offset] << 8) | chunk[offset + 1];
                offset += 2;

                auto entry = unpackEntry(packed);
                record_entry(entry, false);

                if (num_plies > 0) {
                    PackedMoveScoreListReader movelist_reader(
                        entry,
                        reinterpret_cast<unsigned char*>(chunk.data()) + offset,
                        num_plies
                    );
                    while (movelist_reader.hasNext()) {
                        if (sample_limit_reached()) {
                            reached_sample_limit = true;
                            break;
                        }
                        entry = movelist_reader.nextEntry();
                        record_entry(entry, true);
                    }
                    offset += movelist_reader.numReadBytes();
                    if (reached_sample_limit) {
                        break;
                    }
                }
            }
            if (reached_sample_limit) {
                break;
            }
        }

        if (stats.chunk_count == 0) {
            stats.min_chunk_payload_bytes = 0;
        }
        if (stats.entries_read == 0) {
            stats.min_score = 0;
            stats.max_score = 0;
            stats.min_ply = 0;
            stats.max_ply = 0;
        } else {
            stats.min_score = score_to_cp(static_cast<double>(min_score_raw));
            stats.max_score = score_to_cp(static_cast<double>(max_score_raw));
            stats.mean_raw_score = total_raw_score / static_cast<double>(stats.entries_read);
            stats.raw_score_std = population_std(total_raw_score, total_raw_score_sq, stats.entries_read);
            stats.mean_score = total_score / static_cast<double>(stats.entries_read);
            stats.score_std = population_std(total_score, total_score_sq, stats.entries_read);
            stats.mean_abs_score = total_abs_score / static_cast<double>(stats.entries_read);
            stats.abs_score_std = population_std(total_abs_score, total_abs_score_sq, stats.entries_read);
            stats.mean_ply = total_ply / static_cast<double>(stats.entries_read);
            stats.ply_std = population_std(total_ply, total_ply_sq, stats.entries_read);
            stats.mean_piece_count = total_piece_count / static_cast<double>(stats.entries_read);
            stats.piece_count_std = population_std(total_piece_count, total_piece_count_sq, stats.entries_read);
            stats.mean_non_king_piece_count = total_non_king_piece_count / static_cast<double>(stats.entries_read);
            stats.result_mean = total_result / static_cast<double>(stats.entries_read);
            stats.result_wdl_mean = total_result_wdl / static_cast<double>(stats.entries_read);

            const double score_mean = stats.mean_score;
            const double result_mean = stats.result_mean;
            const double score_variance = (total_score_sq / static_cast<double>(stats.entries_read)) - (score_mean * score_mean);
            const double result_variance = (total_result_sq / static_cast<double>(stats.entries_read)) - (result_mean * result_mean);
            const double covariance = (total_score_result / static_cast<double>(stats.entries_read)) - (score_mean * result_mean);
            if (score_variance > 0.0 && result_variance > 0.0) {
                stats.score_result_correlation = covariance / std::sqrt(score_variance * result_variance);
            }

            stats.score_p01 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.01, -32768));
            stats.score_p05 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.05, -32768));
            stats.score_p10 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.10, -32768));
            stats.score_p25 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.25, -32768));
            stats.score_p50 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.50, -32768));
            stats.score_p75 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.75, -32768));
            stats.score_p90 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.90, -32768));
            stats.score_p95 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.95, -32768));
            stats.score_p99 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.99, -32768));
            stats.score_p999 = score_to_cp(percentile_from_histogram(*score_histogram, stats.entries_read, 0.999, -32768));
            stats.abs_score_p50 = score_to_cp(percentile_from_histogram(*abs_score_histogram, stats.entries_read, 0.50, 0));
            stats.abs_score_p75 = score_to_cp(percentile_from_histogram(*abs_score_histogram, stats.entries_read, 0.75, 0));
            stats.abs_score_p90 = score_to_cp(percentile_from_histogram(*abs_score_histogram, stats.entries_read, 0.90, 0));
            stats.abs_score_p95 = score_to_cp(percentile_from_histogram(*abs_score_histogram, stats.entries_read, 0.95, 0));
            stats.abs_score_p99 = score_to_cp(percentile_from_histogram(*abs_score_histogram, stats.entries_read, 0.99, 0));
            stats.abs_score_p999 = score_to_cp(percentile_from_histogram(*abs_score_histogram, stats.entries_read, 0.999, 0));
            stats.ply_p05 = percentile_from_histogram(*ply_histogram, stats.entries_read, 0.05, 0);
            stats.ply_p25 = percentile_from_histogram(*ply_histogram, stats.entries_read, 0.25, 0);
            stats.ply_p50 = percentile_from_histogram(*ply_histogram, stats.entries_read, 0.50, 0);
            stats.ply_p75 = percentile_from_histogram(*ply_histogram, stats.entries_read, 0.75, 0);
            stats.ply_p95 = percentile_from_histogram(*ply_histogram, stats.entries_read, 0.95, 0);
            stats.ply_p99 = percentile_from_histogram(*ply_histogram, stats.entries_read, 0.99, 0);
            stats.piece_count_p05 = percentile_from_histogram(piece_count_histogram, stats.entries_read, 0.05, 0);
            stats.piece_count_p25 = percentile_from_histogram(piece_count_histogram, stats.entries_read, 0.25, 0);
            stats.piece_count_p50 = percentile_from_histogram(piece_count_histogram, stats.entries_read, 0.50, 0);
            stats.piece_count_p75 = percentile_from_histogram(piece_count_histogram, stats.entries_read, 0.75, 0);
            stats.piece_count_p95 = percentile_from_histogram(piece_count_histogram, stats.entries_read, 0.95, 0);

            for (std::size_t i = 0; i < 4; ++i) {
                if (stats.phase_counts[i] == 0) {
                    continue;
                }
                const auto count = static_cast<double>(stats.phase_counts[i]);
                stats.phase_mean_score[i] = phase_score_sums[i] / count;
                stats.phase_mean_abs_score[i] = phase_abs_score_sums[i] / count;
                stats.phase_result_mean[i] = phase_result_sums[i] / count;
            }
            if (result_counts[0] > 0) {
                stats.mean_score_win = result_score_sums[0] / static_cast<double>(result_counts[0]);
                stats.mean_abs_score_win = result_abs_score_sums[0] / static_cast<double>(result_counts[0]);
            }
            if (result_counts[1] > 0) {
                stats.mean_score_draw = result_score_sums[1] / static_cast<double>(result_counts[1]);
                stats.mean_abs_score_draw = result_abs_score_sums[1] / static_cast<double>(result_counts[1]);
            }
            if (result_counts[2] > 0) {
                stats.mean_score_loss = result_score_sums[2] / static_cast<double>(result_counts[2]);
                stats.mean_abs_score_loss = result_abs_score_sums[2] / static_cast<double>(result_counts[2]);
            }
        }

        *out_stats = stats;
        return 1;
    } catch (const std::exception& ex) {
        store_error(ex);
        return 0;
    }
}

extern "C" std::int32_t thrawn_inspect_binpack(const char* path, ThrawnInspectStats* out_stats) {
    return inspect_binpack_impl(path, out_stats, BinpackFilterOptions{});
}

extern "C" std::int32_t thrawn_inspect_binpack_with_options(
    const char* path,
    ThrawnInspectStats* out_stats,
    std::int32_t skip_wdl_score_mismatch,
    std::uint64_t sample_entries
) {
    BinpackFilterOptions filter_options;
    filter_options.skip_wdl_score_mismatch = skip_wdl_score_mismatch != 0;
    filter_options.sample_entries = sample_entries;
    return inspect_binpack_impl(path, out_stats, filter_options);
}

extern "C" std::int32_t thrawn_write_fixture_binpack(const char* path) {
    clear_error();
    try {
        if (path == nullptr) {
            throw std::runtime_error("fixture path must not be null");
        }

        CompressedTrainingDataEntryWriter writer(path, std::ios_base::binary | std::ios_base::trunc);
        writer.addTrainingDataEntry(make_entry(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e2e4",
            24,
            1,
            1
        ));
        writer.addTrainingDataEntry(make_entry(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "g1f3",
            31,
            2,
            1
        ));
        writer.addTrainingDataEntry(make_entry(
            "8/8/8/8/8/8/8/K6k b - - 0 1",
            "h1g1",
            -12,
            1,
            -1
        ));
        return 1;
    } catch (const std::exception& ex) {
        store_error(ex);
        return 0;
    }
}

extern "C" const char* thrawn_last_error() {
    return g_last_error.c_str();
}
