#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
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
    std::int32_t* white_counts;
    std::int32_t* black_counts;
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
    std::int16_t min_score;
    std::int16_t max_score;
    std::uint16_t min_ply;
    std::uint16_t max_ply;
    double mean_abs_score;
    double mean_piece_count;
};

void* thrawn_binpack_open_many(const char* const* paths, std::int32_t num_paths, std::int32_t num_threads, std::int32_t cyclic);
void thrawn_binpack_close(void* handle);
ThrawnBatchView* thrawn_binpack_next_batch(void* handle, std::int32_t batch_size);
void thrawn_batch_free(ThrawnBatchView* batch);
std::int32_t thrawn_inspect_binpack(const char* path, ThrawnInspectStats* out_stats);
std::int32_t thrawn_write_fixture_binpack(const char* path);
const char* thrawn_last_error();

}

namespace {

thread_local std::string g_last_error;

constexpr std::int32_t kNumFeatures = 768;
constexpr std::int32_t kMaxActiveFeatures = 32;

struct ReaderHandle {
    explicit ReaderHandle(
        std::vector<std::string> input_paths,
        std::int32_t num_threads,
        bool cyclic
    ) :
        paths(std::move(input_paths)),
        reader(std::make_unique<CompressedTrainingDataEntryParallelReader>(
            std::max(1, num_threads),
            paths,
            std::ios_base::binary,
            cyclic
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

[[nodiscard]] std::int32_t feature_index(chess::Color perspective, chess::Square sq, chess::Piece piece) {
    const auto piece_bucket =
        static_cast<std::int32_t>(piece.type()) * 2 +
        static_cast<std::int32_t>(piece.color() != perspective);
    return piece_bucket * 64 + orient_square(perspective, sq);
}

[[nodiscard]] std::int32_t fill_feature_list(
    const chess::Position& pos,
    chess::Color perspective,
    std::int32_t* out_features
) {
    std::int32_t count = 0;
    for (chess::Square sq : pos.piecesBB()) {
        const auto piece = pos.pieceAt(sq);
        if (piece == chess::Piece::none()) {
            continue;
        }
        if (count >= kMaxActiveFeatures) {
            break;
        }
        out_features[count++] = feature_index(perspective, sq, piece);
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
    std::int32_t cyclic
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

        auto* handle = new ReaderHandle(std::move(owned_paths), num_threads, cyclic != 0);
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
        batch->white_counts = alloc_array<std::int32_t>(filled);
        batch->black_counts = alloc_array<std::int32_t>(filled);
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

            batch->white_counts[i] = fill_feature_list(
                entry.pos,
                chess::Color::White,
                batch->white_indices + base
            );
            batch->black_counts[i] = fill_feature_list(
                entry.pos,
                chess::Color::Black,
                batch->black_indices + base
            );
            batch->stm[i] = entry.pos.sideToMove() == chess::Color::White ? 1.0f : 0.0f;
            batch->score_cp[i] = static_cast<float>(entry.score);
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
    delete[] batch->white_counts;
    delete[] batch->black_counts;
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
        stats.min_score = std::numeric_limits<std::int16_t>::max();
        stats.max_score = std::numeric_limits<std::int16_t>::min();
        stats.min_ply = std::numeric_limits<std::uint16_t>::max();
        stats.max_ply = std::numeric_limits<std::uint16_t>::min();

        double total_abs_score = 0.0;
        double total_piece_count = 0.0;

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
            } else if (entry.result < 0) {
                ++stats.losses;
            } else {
                ++stats.draws;
            }

            stats.min_score = std::min(stats.min_score, entry.score);
            stats.max_score = std::max(stats.max_score, entry.score);
            stats.min_ply = std::min(stats.min_ply, entry.ply);
            stats.max_ply = std::max(stats.max_ply, entry.ply);

            total_abs_score += std::abs(static_cast<double>(entry.score));
            total_piece_count += static_cast<double>(entry.pos.piecesBB().count());
        }

        if (stats.entries_read == 0) {
            stats.min_score = 0;
            stats.max_score = 0;
            stats.min_ply = 0;
            stats.max_ply = 0;
        } else {
            stats.mean_abs_score = total_abs_score / static_cast<double>(stats.entries_read);
            stats.mean_piece_count = total_piece_count / static_cast<double>(stats.entries_read);
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
