// Generate reproducible, uniformly random legal Othello games for board probing.
// Build: clang++ -O3 -std=c++17 scripts/generate_probe_games.cpp -o /tmp/generate_probe_games
// Matches data/othello.py's implicit-pass rules and the released probe's parity labels.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Bits = std::uint64_t;
constexpr Bits not_a = 0xfefefefefefefefeULL;
constexpr Bits not_h = 0x7f7f7f7f7f7f7f7fULL;
constexpr std::size_t moves_per_game = 60;
constexpr std::size_t label_positions = 59;
constexpr std::size_t squares = 64;

Bits shift(Bits x, int direction) {
    switch (direction) {
        case 0: return x >> 8;
        case 1: return (x & not_h) >> 7;
        case 2: return (x & not_h) << 1;
        case 3: return (x & not_h) << 9;
        case 4: return x << 8;
        case 5: return (x & not_a) << 7;
        case 6: return (x & not_a) >> 1;
        case 7: return (x & not_a) >> 9;
    }
    throw std::logic_error("Invalid direction");
}

Bits legal_moves(Bits player, Bits opponent) {
    const Bits empty = ~(player | opponent);
    Bits result = 0;
    for (int d = 0; d < 8; ++d) {
        Bits ray = shift(player, d) & opponent;
        // At most six opponent discs fit between a player disc and an empty cell.
        for (int step = 0; step < 5; ++step) ray |= shift(ray, d) & opponent;
        result |= shift(ray, d) & empty;
    }
    return result;
}

Bits flips(Bits move, Bits player, Bits opponent) {
    Bits result = 0;
    for (int d = 0; d < 8; ++d) {
        Bits ray = shift(move, d) & opponent;
        Bits captured = 0;
        while (ray) {
            captured |= ray;
            const Bits next = shift(ray, d);
            if (next & player) {
                result |= captured;
                break;
            }
            ray = next & opponent;
        }
    }
    return result;
}

// Explicit rejection sampling avoids implementation-dependent uniform_int_distribution.
std::uint64_t uniform_below(std::mt19937_64& rng, std::uint64_t bound) {
    const std::uint64_t threshold = (std::uint64_t(0) - bound) % bound;
    std::uint64_t value;
    do { value = rng(); } while (value < threshold);
    return value % bound;
}

struct Game {
    std::array<std::uint8_t, moves_per_game> moves{};
    std::array<std::uint8_t, label_positions * squares> labels{};
    std::size_t length = 0;
    std::uint64_t passes = 0;
};

Game play_game(std::mt19937_64& rng) {
    Bits black = (Bits(1) << 28) | (Bits(1) << 35);
    Bits white = (Bits(1) << 27) | (Bits(1) << 36);
    bool black_to_move = true;
    Game game;
    while (game.length < moves_per_game) {
        Bits player = black_to_move ? black : white;
        Bits opponent = black_to_move ? white : black;
        Bits options = legal_moves(player, opponent);
        if (!options) {
            options = legal_moves(opponent, player);
            if (!options) break;
            black_to_move = !black_to_move;
            std::swap(player, opponent);
            ++game.passes;
        }
        std::uint64_t rank = uniform_below(rng, __builtin_popcountll(options));
        while (rank--) options &= options - 1;
        const unsigned int cell = __builtin_ctzll(options);
        const Bits move = Bits(1) << cell;
        const Bits captured = flips(move, player, opponent);
        if (!captured) throw std::logic_error("A selected legal move flips no discs");
        player |= move | captured;
        opponent &= ~captured;
        if (black_to_move) { black = player; white = opponent; }
        else { white = player; black = opponent; }
        const std::size_t t = game.length;
        game.moves[t] = static_cast<std::uint8_t>(cell);
        if (t < label_positions) {
            for (std::size_t square = 0; square < squares; ++square) {
                const Bits bit = Bits(1) << square;
                // Released state_stack_to_one_hot_threeway uses token parity,
                // even when actual player alternation changes after a pass.
                game.labels[t * squares + square] = (black & bit) ? (t % 2 ? 2 : 1)
                    : (white & bit) ? (t % 2 ? 1 : 2) : 0;
            }
        }
        ++game.length;
        black_to_move = !black_to_move;
    }
    return game;
}

struct Options {
    fs::path output_directory;
    std::uint64_t train_games = 100000;
    std::uint64_t val_games = 512;
    std::uint64_t test_games = 1000;
    std::uint64_t seed = 42;
};

std::uint64_t parse_unsigned(const std::string& value) {
    if (value.empty() || value.front() == '-') throw std::invalid_argument("Expected a nonnegative integer");
    std::size_t consumed = 0;
    const std::uint64_t number = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("Expected an integer: " + value);
    return number;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag(argv[i]);
        if (flag == "--help" || flag == "-h") {
            std::cout << "Usage: generate_probe_games --output-directory PATH [--train-games 100000] "
                         "[--val-games 512] [--test-games 1000] [--seed 42]\n";
            std::exit(0);
        }
        if (++i == argc) throw std::invalid_argument("Missing value for " + flag);
        const std::string value(argv[i]);
        if (flag == "--output-directory") options.output_directory = value;
        else if (flag == "--train-games") options.train_games = parse_unsigned(value);
        else if (flag == "--val-games") options.val_games = parse_unsigned(value);
        else if (flag == "--test-games") options.test_games = parse_unsigned(value);
        else if (flag == "--seed") options.seed = parse_unsigned(value);
        else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (options.output_directory.empty()) throw std::invalid_argument("--output-directory is required");
    return options;
}

struct Split {
    std::string name;
    std::uint64_t requested;
    std::uint64_t generated = 0;
    std::uint64_t rejected_short = 0;
    std::uint64_t rejected_duplicate = 0;
    std::uint64_t retained_passes = 0;
    std::uint64_t retained_games_with_passes = 0;
    std::uint64_t attempted_passes = 0;
    std::uint64_t rng_game_offset = 0;
};

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::vector<Split> splits{{"train", options.train_games}, {"val", options.val_games},
                                  {"test", options.test_games}};
        fs::create_directories(options.output_directory);
        // Prevent inadvertently overwriting a partially or fully generated dataset.
        for (const auto& split : splits) {
            for (const auto& kind : {"moves", "labels"}) {
                const fs::path path = options.output_directory / (split.name + "_" + kind + ".uint8");
                if (fs::exists(path)) throw std::runtime_error("Output already exists: " + path.string());
            }
        }
        const fs::path manifest_path = options.output_directory / "dataset_manifest.json";
        if (fs::exists(manifest_path)) throw std::runtime_error("Manifest already exists: " + manifest_path.string());
        std::mt19937_64 rng(options.seed);
        std::unordered_set<std::string> seen;
        seen.reserve(options.train_games + options.val_games + options.test_games);
        std::uint64_t attempted_total = 0;
        const auto start = std::chrono::steady_clock::now();
        for (auto& split : splits) {
            split.rng_game_offset = attempted_total;
            std::ofstream moves_file(options.output_directory / (split.name + "_moves.uint8"), std::ios::binary);
            std::ofstream labels_file(options.output_directory / (split.name + "_labels.uint8"), std::ios::binary);
            if (!moves_file || !labels_file) throw std::runtime_error("Cannot open output files");
            while (split.generated < split.requested) {
                const Game game = play_game(rng);
                ++attempted_total;
                split.attempted_passes += game.passes;
                if (game.length != moves_per_game) { ++split.rejected_short; continue; }
                const std::string identity(reinterpret_cast<const char*>(game.moves.data()), moves_per_game);
                if (!seen.insert(identity).second) { ++split.rejected_duplicate; continue; }
                moves_file.write(reinterpret_cast<const char*>(game.moves.data()), game.moves.size());
                labels_file.write(reinterpret_cast<const char*>(game.labels.data()), game.labels.size());
                if (!moves_file || !labels_file) throw std::runtime_error("Writing dataset failed");
                ++split.generated;
                split.retained_passes += game.passes;
                split.retained_games_with_passes += game.passes > 0;
                if (split.generated % 10000 == 0) {
                    std::cerr << split.name << ": " << split.generated << '/' << split.requested << " retained\n";
                }
            }
            moves_file.flush();
            labels_file.flush();
            if (!moves_file || !labels_file) throw std::runtime_error("Flushing dataset failed");
            std::cerr << split.name << ": finished " << split.generated << " games; rejected "
                      << split.rejected_short << " short and " << split.rejected_duplicate << " duplicate games\n";
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream manifest(manifest_path);
        manifest << "{\n  \"schema_version\": 1,\n  \"seed\": " << options.seed
                 << ",\n  \"data_identity\": \"seeded_regeneration_not_authors_original_game_ids\",\n"
                    "  \"algorithm\": \"uniform random legal moves; implicit forced passes; retain length 60; exact full-sequence deduplication across splits\",\n"
                    "  \"random_generator\": \"std::mt19937_64; unbiased integer rejection sampling\",\n"
                    "  \"source_engine\": \"https://github.com/likenneth/othello_world/blob/master/data/othello.py\",\n"
                    "  \"source_filter\": \"https://github.com/likenneth/othello_world/blob/master/mechanistic_interpretability/tl_initial_exploration.py#L20-L58\",\n"
                    "  \"source_labels\": \"mech_int/tl_othello_utils.py:state_stack_to_one_hot_threeway\",\n"
                    "  \"move_encoding\": \"row-major board index 0..63; not GPT vocabulary token\",\n"
                    "  \"labels\": [\"empty\", \"yours\", \"mine\"],\n"
                    "  \"label_semantics\": \"board after each move; token-index parity: even black=yours, white=mine; odd reversed, including after forced passes\",\n"
                    "  \"initial_board\": {\"black\": [28,35], \"white\": [27,36]},\n"
                    "  \"split_order\": [\"train\", \"val\", \"test\"],\n"
                    "  \"attempted_games_total\": " << attempted_total << ",\n"
                    "  \"elapsed_seconds\": " << std::setprecision(10) << elapsed << ",\n  \"splits\": {\n";
        for (std::size_t i = 0; i < splits.size(); ++i) {
            const auto& split = splits[i];
            manifest << "    \"" << split.name << "\": {\n"
                     << "      \"games\": " << split.generated << ",\n"
                     << "      \"attempted_games\": " << split.generated + split.rejected_short + split.rejected_duplicate << ",\n"
                     << "      \"rejected_short\": " << split.rejected_short << ",\n"
                     << "      \"rejected_duplicate\": " << split.rejected_duplicate << ",\n"
                     << "      \"retained_passes\": " << split.retained_passes << ",\n"
                     << "      \"retained_games_with_passes\": " << split.retained_games_with_passes << ",\n"
                     << "      \"attempted_passes\": " << split.attempted_passes << ",\n"
                     << "      \"rng_game_offset\": " << split.rng_game_offset << ",\n"
                     << "      \"moves_file\": \"" << split.name << "_moves.uint8\",\n"
                     << "      \"moves_dtype\": \"uint8\",\n"
                     << "      \"moves_shape\": [" << split.generated << ",60],\n"
                     << "      \"labels_file\": \"" << split.name << "_labels.uint8\",\n"
                     << "      \"labels_dtype\": \"uint8\",\n"
                     << "      \"labels_shape\": [" << split.generated << ",59,64]\n"
                     << "    }" << (i + 1 == splits.size() ? "\n" : ",\n");
        }
        manifest << "  }\n}\n";
        manifest.flush();
        if (!manifest) throw std::runtime_error("Writing manifest failed");
        std::cout << "Generated " << seen.size() << " disjoint full-length games in " << elapsed << " seconds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
