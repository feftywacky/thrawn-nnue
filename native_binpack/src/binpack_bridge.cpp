#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <array>
#include <vector>

#include "../third_party/nnue_pytorch/nnue_training_data_formats.h"

using binpack::CompressedTrainingDataEntryParallelReader;
using binpack::CompressedTrainingDataEntryReader;
using binpack::CompressedTrainingDataEntryWriter;
using binpack::TrainingDataEntry;

extern "C" {

struct ThrawnBatchView {
    std::int32_t size;
    std::int32_t max_active_features;
    std::int32_t* white_indices;
    std::int32_t* black_indices;
    float* stm;
    float* score_cp;
    float* result_wdl;
};

struct ThrawnInspectStats {
    std::uint64_t entries_read;
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
    double mean_abs_score;
    double abs_score_std;
    double mean_ply;
    double ply_std;
    double mean_piece_count;
    double piece_count_std;
    double mean_non_king_piece_count;
    double result_mean;
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
    std::uint64_t piece_type_counts[6];
    std::uint64_t white_piece_counts[6];
    std::uint64_t black_piece_counts[6];
    double wdl_scale_signed_target_mean[7];
    double wdl_scale_signed_target_std[7];
    double wdl_scale_abs_target_mean[7];
    std::uint64_t wdl_scale_saturated_99[7];
    std::uint64_t wdl_scale_saturated_999[7];
};

void* thrawn_binpack_open_many(
    const char* const* paths,
    std::int32_t num_paths,
    std::int32_t num_threads,
    std::int32_t cyclic,
    std::int32_t skip_capture_positions,
    std::int32_t skip_decisive_score_mismatch,
    double decisive_score_mismatch_margin,
    std::int32_t skip_draw_score_mismatch,
    double draw_score_mismatch_margin,
    double max_abs_score
);
void thrawn_binpack_close(void* handle);
ThrawnBatchView* thrawn_binpack_next_batch(void* handle, std::int32_t batch_size);
void thrawn_batch_free(ThrawnBatchView* batch);
std::int32_t thrawn_inspect_binpack(const char* path, ThrawnInspectStats* out_stats);
std::int32_t thrawn_write_fixture_binpack(const char* path);
const char* thrawn_last_error();

}

namespace {

thread_local std::string g_last_error;

constexpr std::int32_t kFactorFeatures = 640;
constexpr std::int32_t kNumFeatures = 40960;
constexpr std::int32_t kMaxActiveFeatures = 30;
constexpr double kStockfishInternalToCp = 100.0 / 208.0;
constexpr std::array<double, 7> kWdlScales{197.0, 410.0, 600.0, 1000.0, 2000.0, 4000.0, 8000.0};
constexpr std::array<chess::PieceType, 6> kPieceTypes{
    chess::PieceType::Pawn,
    chess::PieceType::Knight,
    chess::PieceType::Bishop,
    chess::PieceType::Rook,
    chess::PieceType::Queen,
    chess::PieceType::King,
};

struct BinpackFilterOptions {
    bool skip_capture_positions = false;
    bool skip_decisive_score_mismatch = false;
    double decisive_score_mismatch_margin = 0.0;
    bool skip_draw_score_mismatch = false;
    double draw_score_mismatch_margin = 0.0;
    double max_abs_score = 0.0;
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

[[nodiscard]] std::int32_t orient_square(chess::Color perspective, chess::Square sq) {
    if (perspective == chess::Color::White) {
        return static_cast<std::int32_t>(sq);
    }

    const auto file = sq.file();
    const auto rank = static_cast<chess::Rank>(7 - static_cast<int>(sq.rank()));
    return static_cast<std::int32_t>(chess::Square(file, rank));
}

[[nodiscard]] std::int32_t factor_feature_index(chess::Color perspective, chess::Square sq, chess::Piece piece) {
    const auto piece_bucket =
        static_cast<std::int32_t>(piece.type()) * 2 +
        static_cast<std::int32_t>(piece.color() != perspective);
    return piece_bucket * 64 + orient_square(perspective, sq);
}

[[nodiscard]] std::int32_t feature_index(
    chess::Color perspective,
    chess::Square king_sq,
    chess::Square sq,
    chess::Piece piece
) {
    const auto oriented_king = orient_square(perspective, king_sq);
    return oriented_king * kFactorFeatures + factor_feature_index(perspective, sq, piece);
}

[[nodiscard]] std::int32_t fill_feature_list(
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
        if (piece.type() == chess::PieceType::King) {
            continue;
        }
        if (count >= kMaxActiveFeatures) {
            break;
        }
        out_features[count++] = feature_index(perspective, king_sq, sq, piece);
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

[[nodiscard]] bool should_skip_entry(const TrainingDataEntry& entry, const BinpackFilterOptions& options) {
    if (options.skip_capture_positions && entry.isCapturingMove()) {
        return true;
    }

    const double score_cp = score_to_cp(static_cast<double>(entry.score));
    const double abs_score_cp = std::abs(score_cp);

    if (options.max_abs_score > 0.0 && abs_score_cp > options.max_abs_score) {
        return true;
    }

    if (options.skip_decisive_score_mismatch && options.decisive_score_mismatch_margin > 0.0) {
        if (entry.result > 0 && score_cp <= -options.decisive_score_mismatch_margin) {
            return true;
        }
        if (entry.result < 0 && score_cp >= options.decisive_score_mismatch_margin) {
            return true;
        }
    }

    if (
        options.skip_draw_score_mismatch &&
        options.draw_score_mismatch_margin > 0.0 &&
        entry.result == 0 &&
        abs_score_cp >= options.draw_score_mismatch_margin
    ) {
        return true;
    }

    return false;
}

[[nodiscard]] std::function<bool(const TrainingDataEntry&)> make_skip_predicate(BinpackFilterOptions options) {
    return [options](const TrainingDataEntry& entry) {
        return should_skip_entry(entry, options);
    };
}

[[nodiscard]] double wdl_target(double score_cp, double scale) {
    return 1.0 / (1.0 + std::exp(-(score_cp / scale)));
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
    std::int32_t skip_capture_positions,
    std::int32_t skip_decisive_score_mismatch,
    double decisive_score_mismatch_margin,
    std::int32_t skip_draw_score_mismatch,
    double draw_score_mismatch_margin,
    double max_abs_score
) {
    clear_error();
    try {
        if (num_paths <= 0) {
            throw std::runtime_error("thrawn_binpack_open_many requires at least one dataset path");
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
        filter_options.skip_capture_positions = skip_capture_positions != 0;
        filter_options.skip_decisive_score_mismatch = skip_decisive_score_mismatch != 0;
        filter_options.decisive_score_mismatch_margin = decisive_score_mismatch_margin;
        filter_options.skip_draw_score_mismatch = skip_draw_score_mismatch != 0;
        filter_options.draw_score_mismatch_margin = draw_score_mismatch_margin;
        filter_options.max_abs_score = max_abs_score;

        auto* handle = new ReaderHandle(std::move(owned_paths), num_threads, cyclic != 0, filter_options);
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
        batch->size = filled;
        batch->max_active_features = kMaxActiveFeatures;
        batch->white_indices = alloc_array<std::int32_t>(static_cast<std::size_t>(filled) * kMaxActiveFeatures);
        batch->black_indices = alloc_array<std::int32_t>(static_cast<std::size_t>(filled) * kMaxActiveFeatures);
        batch->stm = alloc_array<float>(filled);
        batch->score_cp = alloc_array<float>(filled);
        batch->result_wdl = alloc_array<float>(filled);

        std::fill(
            batch->white_indices,
            batch->white_indices + static_cast<std::size_t>(filled) * kMaxActiveFeatures,
            -1
        );
        std::fill(
            batch->black_indices,
            batch->black_indices + static_cast<std::size_t>(filled) * kMaxActiveFeatures,
            -1
        );

        for (int i = 0; i < filled; ++i) {
            const auto& entry = entries[static_cast<std::size_t>(i)];
            const auto base = static_cast<std::size_t>(i) * kMaxActiveFeatures;

            static_cast<void>(fill_feature_list(
                entry.pos,
                chess::Color::White,
                batch->white_indices + base
            ));
            static_cast<void>(fill_feature_list(
                entry.pos,
                chess::Color::Black,
                batch->black_indices + base
            ));
            batch->stm[i] = entry.pos.sideToMove() == chess::Color::White ? 1.0f : 0.0f;
            batch->score_cp[i] = static_cast<float>(score_to_cp(static_cast<double>(entry.score)));
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
    delete[] batch->score_cp;
    delete[] batch->result_wdl;
    delete batch;
}

extern "C" std::int32_t thrawn_inspect_binpack(const char* path, ThrawnInspectStats* out_stats) {
    clear_error();
    try {
        if (path == nullptr) {
            throw std::runtime_error("inspect path must not be null");
        }
        if (out_stats == nullptr) {
            throw std::runtime_error("inspect output struct must not be null");
        }

        CompressedTrainingDataEntryReader reader(path, std::ios_base::binary);

        ThrawnInspectStats stats{};
        std::int16_t min_score_raw = std::numeric_limits<std::int16_t>::max();
        std::int16_t max_score_raw = std::numeric_limits<std::int16_t>::min();
        stats.min_ply = std::numeric_limits<std::uint16_t>::max();
        stats.max_ply = std::numeric_limits<std::uint16_t>::min();

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
        double total_score_result = 0.0;
        std::array<std::uint64_t, 65536> score_histogram{};
        std::array<std::uint64_t, 32769> abs_score_histogram{};
        std::array<std::uint64_t, 65536> ply_histogram{};
        std::array<std::uint64_t, 33> piece_count_histogram{};
        std::array<double, 4> phase_score_sums{};
        std::array<double, 4> phase_abs_score_sums{};
        std::array<double, 4> phase_result_sums{};
        std::array<std::uint64_t, 3> result_counts{};
        std::array<double, 3> result_score_sums{};
        std::array<double, 3> result_abs_score_sums{};

        while (reader.hasNext()) {
            const auto entry = reader.next();
            ++stats.entries_read;
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
            total_score_result += score_cp * result_value;
            score_histogram[static_cast<std::uint16_t>(static_cast<int>(entry.score) + 32768)] += 1;
            abs_score_histogram[std::min(abs_score, 32768)] += 1;
            ply_histogram[entry.ply] += 1;
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

        }

        if (stats.entries_read == 0) {
            stats.min_score = 0;
            stats.max_score = 0;
            stats.min_ply = 0;
            stats.max_ply = 0;
        } else {
            stats.min_score = score_to_cp(static_cast<double>(min_score_raw));
            stats.max_score = score_to_cp(static_cast<double>(max_score_raw));
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

            const double score_mean = stats.mean_score;
            const double result_mean = stats.result_mean;
            const double score_variance = (total_score_sq / static_cast<double>(stats.entries_read)) - (score_mean * score_mean);
            const double result_variance = (total_result_sq / static_cast<double>(stats.entries_read)) - (result_mean * result_mean);
            const double covariance = (total_score_result / static_cast<double>(stats.entries_read)) - (score_mean * result_mean);
            if (score_variance > 0.0 && result_variance > 0.0) {
                stats.score_result_correlation = covariance / std::sqrt(score_variance * result_variance);
            }

            stats.score_p01 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.01, -32768));
            stats.score_p05 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.05, -32768));
            stats.score_p10 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.10, -32768));
            stats.score_p25 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.25, -32768));
            stats.score_p50 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.50, -32768));
            stats.score_p75 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.75, -32768));
            stats.score_p90 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.90, -32768));
            stats.score_p95 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.95, -32768));
            stats.score_p99 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.99, -32768));
            stats.score_p999 = score_to_cp(percentile_from_histogram(score_histogram, stats.entries_read, 0.999, -32768));
            stats.abs_score_p50 = score_to_cp(percentile_from_histogram(abs_score_histogram, stats.entries_read, 0.50, 0));
            stats.abs_score_p75 = score_to_cp(percentile_from_histogram(abs_score_histogram, stats.entries_read, 0.75, 0));
            stats.abs_score_p90 = score_to_cp(percentile_from_histogram(abs_score_histogram, stats.entries_read, 0.90, 0));
            stats.abs_score_p95 = score_to_cp(percentile_from_histogram(abs_score_histogram, stats.entries_read, 0.95, 0));
            stats.abs_score_p99 = score_to_cp(percentile_from_histogram(abs_score_histogram, stats.entries_read, 0.99, 0));
            stats.abs_score_p999 = score_to_cp(percentile_from_histogram(abs_score_histogram, stats.entries_read, 0.999, 0));
            stats.ply_p05 = percentile_from_histogram(ply_histogram, stats.entries_read, 0.05, 0);
            stats.ply_p25 = percentile_from_histogram(ply_histogram, stats.entries_read, 0.25, 0);
            stats.ply_p50 = percentile_from_histogram(ply_histogram, stats.entries_read, 0.50, 0);
            stats.ply_p75 = percentile_from_histogram(ply_histogram, stats.entries_read, 0.75, 0);
            stats.ply_p95 = percentile_from_histogram(ply_histogram, stats.entries_read, 0.95, 0);
            stats.ply_p99 = percentile_from_histogram(ply_histogram, stats.entries_read, 0.99, 0);
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

            std::array<double, 7> wdl_signed_sums{};
            std::array<double, 7> wdl_signed_sq_sums{};
            std::array<double, 7> wdl_abs_sums{};
            for (std::size_t score_index = 0; score_index < score_histogram.size(); ++score_index) {
                const auto count = score_histogram[score_index];
                if (count == 0) {
                    continue;
                }
                const double score_cp = score_to_cp(static_cast<double>(static_cast<int>(score_index) - 32768));
                for (std::size_t scale_index = 0; scale_index < kWdlScales.size(); ++scale_index) {
                    const double signed_target = wdl_target(score_cp, kWdlScales[scale_index]);
                    wdl_signed_sums[scale_index] += signed_target * static_cast<double>(count);
                    wdl_signed_sq_sums[scale_index] += signed_target * signed_target * static_cast<double>(count);
                    stats.wdl_scale_saturated_99[scale_index] +=
                        (signed_target <= 0.01 || signed_target >= 0.99) ? count : 0;
                    stats.wdl_scale_saturated_999[scale_index] +=
                        (signed_target <= 0.001 || signed_target >= 0.999) ? count : 0;
                }
            }
            for (std::size_t abs_index = 0; abs_index < abs_score_histogram.size(); ++abs_index) {
                const auto count = abs_score_histogram[abs_index];
                if (count == 0) {
                    continue;
                }
                const double abs_score_cp = score_to_cp(static_cast<double>(abs_index));
                for (std::size_t scale_index = 0; scale_index < kWdlScales.size(); ++scale_index) {
                    const double abs_target = wdl_target(abs_score_cp, kWdlScales[scale_index]);
                    wdl_abs_sums[scale_index] += abs_target * static_cast<double>(count);
                }
            }
            for (std::size_t i = 0; i < kWdlScales.size(); ++i) {
                stats.wdl_scale_signed_target_mean[i] = wdl_signed_sums[i] / static_cast<double>(stats.entries_read);
                stats.wdl_scale_signed_target_std[i] = population_std(
                    wdl_signed_sums[i],
                    wdl_signed_sq_sums[i],
                    stats.entries_read
                );
                stats.wdl_scale_abs_target_mean[i] = wdl_abs_sums[i] / static_cast<double>(stats.entries_read);
            }
        }

        *out_stats = stats;
        return 1;
    } catch (const std::exception& ex) {
        store_error(ex);
        return 0;
    }
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
