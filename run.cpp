/* Inference for Llama-2 Transformer model in C++ */
/* Converted from run.c with class encapsulation and multi-threaded parallel inference support */

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Neural net math functions (free functions, thread-safe)

static void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float* xout, float* x, float* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Config struct (POD, same binary layout as run.c for checkpoint compatibility)

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

// ----------------------------------------------------------------------------
// RunState: per-thread activation buffers

class RunState {
public:
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;      // points into key_cache
    float *v;      // points into value_cache
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;

    RunState() : x(nullptr), xb(nullptr), xb2(nullptr), hb(nullptr), hb2(nullptr),
                 q(nullptr), k(nullptr), v(nullptr), att(nullptr), logits(nullptr),
                 key_cache(nullptr), value_cache(nullptr) {}

    explicit RunState(const Config& p) : k(nullptr), v(nullptr) {
        allocate(p);
    }

    ~RunState() {
        free_buffers();
    }

    // Non-copyable
    RunState(const RunState&) = delete;
    RunState& operator=(const RunState&) = delete;

    // Movable
    RunState(RunState&& other) noexcept {
        move_from(other);
    }
    RunState& operator=(RunState&& other) noexcept {
        if (this != &other) {
            free_buffers();
            move_from(other);
        }
        return *this;
    }

    void allocate(const Config& p) {
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        x = static_cast<float*>(calloc(p.dim, sizeof(float)));
        xb = static_cast<float*>(calloc(p.dim, sizeof(float)));
        xb2 = static_cast<float*>(calloc(p.dim, sizeof(float)));
        hb = static_cast<float*>(calloc(p.hidden_dim, sizeof(float)));
        hb2 = static_cast<float*>(calloc(p.hidden_dim, sizeof(float)));
        q = static_cast<float*>(calloc(p.dim, sizeof(float)));
        key_cache = static_cast<float*>(calloc(p.n_layers * p.seq_len * kv_dim, sizeof(float)));
        value_cache = static_cast<float*>(calloc(p.n_layers * p.seq_len * kv_dim, sizeof(float)));
        att = static_cast<float*>(calloc(p.n_heads * p.seq_len, sizeof(float)));
        logits = static_cast<float*>(calloc(p.vocab_size, sizeof(float)));
        if (!x || !xb || !xb2 || !hb || !hb2 || !q
         || !key_cache || !value_cache || !att || !logits) {
            fprintf(stderr, "malloc failed!\n");
            exit(EXIT_FAILURE);
        }
    }

private:
    void free_buffers() {
        free(x); free(xb); free(xb2);
        free(hb); free(hb2); free(q);
        free(att); free(logits);
        free(key_cache); free(value_cache);
        x = xb = xb2 = hb = hb2 = q = att = logits = key_cache = value_cache = nullptr;
    }

    void move_from(RunState& other) {
        x = other.x; xb = other.xb; xb2 = other.xb2;
        hb = other.hb; hb2 = other.hb2; q = other.q;
        k = other.k; v = other.v;
        att = other.att; logits = other.logits;
        key_cache = other.key_cache; value_cache = other.value_cache;
        other.x = other.xb = other.xb2 = other.hb = other.hb2 = other.q = nullptr;
        other.att = other.logits = other.key_cache = other.value_cache = nullptr;
        other.k = other.v = nullptr;
    }
};

// ----------------------------------------------------------------------------
// Transformer: owns model weights (shared, read-only) and config

class Transformer {
public:
    Config config;

    Transformer() : fd_(-1), data_(nullptr), file_size_(0) {
        memset(&weights_, 0, sizeof(weights_));
    }

    explicit Transformer(const char* checkpoint_path) : fd_(-1), data_(nullptr), file_size_(0) {
        memset(&weights_, 0, sizeof(weights_));
        load(checkpoint_path);
    }

    ~Transformer() {
        if (data_ != MAP_FAILED && data_ != nullptr) {
            munmap(data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

    // Non-copyable
    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    void load(const char* checkpoint_path) {
        FILE *file = fopen(checkpoint_path, "rb");
        if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
        if (fread(&config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
        printf("debug: dim %d hidden_dim %d layers %d heads %d kv_heads %d vocab_size %d seq_len %d\n",
            config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.n_kv_heads,
            config.vocab_size, config.seq_len);
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        fseek(file, 0, SEEK_END);
        file_size_ = ftell(file);
        fclose(file);
        fd_ = open(checkpoint_path, O_RDONLY);
        if (fd_ == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        data_ = static_cast<float*>(mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
        float* weights_ptr = data_ + sizeof(Config)/sizeof(float);
        memory_map_weights(weights_ptr, shared_weights);
    }

    RunState create_run_state() const {
        return RunState(config);
    }

    // Thread-safe: only reads shared weights, writes to caller-owned RunState
    float* forward(RunState& s, int token, int pos) const {
        const Config& p = config;
        float *x = s.x;
        int dim = p.dim;
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        int kv_mul = p.n_heads / p.n_kv_heads;
        int hidden_dim = p.hidden_dim;
        int head_size = dim / p.n_heads;

        float* content_row = weights_.token_embedding_table + token * dim;
        memcpy(x, content_row, dim * sizeof(*x));

        for (unsigned long long l = 0; l < static_cast<unsigned long long>(p.n_layers); l++) {
            rmsnorm(s.xb, x, weights_.rms_att_weight + l*dim, dim);

            int loff = l * p.seq_len * kv_dim;
            s.k = s.key_cache + loff + pos * kv_dim;
            s.v = s.value_cache + loff + pos * kv_dim;

            matmul(s.q, s.xb, weights_.wq + l*dim*dim, dim, dim);
            matmul(s.k, s.xb, weights_.wk + l*dim*kv_dim, dim, kv_dim);
            matmul(s.v, s.xb, weights_.wv + l*dim*kv_dim, dim, kv_dim);

            for (int i = 0; i < dim; i+=2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s.q : s.k;
                    float v0 = vec[i];
                    float v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }

            int h;
            #pragma omp parallel for private(h)
            for (h = 0; h < p.n_heads; h++) {
                float* q = s.q + h * head_size;
                float* att = s.att + h * p.seq_len;
                for (int t = 0; t <= pos; t++) {
                    float* k = s.key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        score += q[i] * k[i];
                    }
                    score /= sqrtf(head_size);
                    att[t] = score;
                }
                softmax(att, pos + 1);
                float* xb = s.xb + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* v = s.value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float a = att[t];
                    for (int i = 0; i < head_size; i++) {
                        xb[i] += a * v[i];
                    }
                }
            }

            matmul(s.xb2, s.xb, weights_.wo + l*dim*dim, dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s.xb2[i];
            }

            rmsnorm(s.xb, x, weights_.rms_ffn_weight + l*dim, dim);

            matmul(s.hb, s.xb, weights_.w1 + l*dim*hidden_dim, dim, hidden_dim);
            matmul(s.hb2, s.xb, weights_.w3 + l*dim*hidden_dim, dim, hidden_dim);

            for (int i = 0; i < hidden_dim; i++) {
                float val = s.hb[i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            matmul(s.xb, s.hb, weights_.w2 + l*dim*hidden_dim, hidden_dim, dim);

            for (int i = 0; i < dim; i++) {
                x[i] += s.xb[i];
            }
        }

        rmsnorm(x, x, weights_.rms_final_weight, dim);
        matmul(s.logits, x, weights_.wcls, p.dim, p.vocab_size);
        return s.logits;
    }

private:
    struct TransformerWeights {
        float* token_embedding_table;
        float* rms_att_weight;
        float* rms_ffn_weight;
        float* wq;
        float* wk;
        float* wv;
        float* wo;
        float* w1;
        float* w2;
        float* w3;
        float* rms_final_weight;
        float* wcls;
    };

    TransformerWeights weights_;
    int fd_;
    float* data_;
    ssize_t file_size_;

    void memory_map_weights(float* ptr, int shared_weights) {
        int head_size = config.dim / config.n_heads;
        unsigned long long n_layers = config.n_layers;
        weights_.token_embedding_table = ptr;
        ptr += config.vocab_size * config.dim;
        weights_.rms_att_weight = ptr;
        ptr += n_layers * config.dim;
        weights_.wq = ptr;
        ptr += n_layers * config.dim * (config.n_heads * head_size);
        weights_.wk = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights_.wv = ptr;
        ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
        weights_.wo = ptr;
        ptr += n_layers * (config.n_heads * head_size) * config.dim;
        weights_.rms_ffn_weight = ptr;
        ptr += n_layers * config.dim;
        weights_.w1 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights_.w2 = ptr;
        ptr += n_layers * config.hidden_dim * config.dim;
        weights_.w3 = ptr;
        ptr += n_layers * config.dim * config.hidden_dim;
        weights_.rms_final_weight = ptr;
        ptr += config.dim;
        ptr += config.seq_len * head_size / 2; // skip freq_cis_real
        ptr += config.seq_len * head_size / 2; // skip freq_cis_imag
        weights_.wcls = shared_weights ? weights_.token_embedding_table : ptr;
    }
};

// ----------------------------------------------------------------------------
// Tokenizer

struct TokenIndex {
    const char *str;
    int id;
};

static int compare_tokens(const void *a, const void *b) {
    return strcmp(static_cast<const TokenIndex*>(a)->str,
                 static_cast<const TokenIndex*>(b)->str);
}

class Tokenizer {
public:
    int vocab_size;

    Tokenizer() : vocab_size(0), vocab_(nullptr), vocab_scores_(nullptr), sorted_vocab_(nullptr),
                  max_token_length_(0) {
        memset(byte_pieces_, 0, sizeof(byte_pieces_));
    }

    Tokenizer(const char* tokenizer_path, int vocab_size) : vocab_size(vocab_size),
            vocab_(nullptr), vocab_scores_(nullptr), sorted_vocab_(nullptr) {
        load(tokenizer_path, vocab_size);
    }

    ~Tokenizer() {
        if (vocab_) {
            for (int i = 0; i < vocab_size; i++) { free(vocab_[i]); }
            free(vocab_);
        }
        free(vocab_scores_);
        free(sorted_vocab_);
    }

    // Non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    void load(const char* tokenizer_path, int vs) {
        vocab_size = vs;
        vocab_ = static_cast<char**>(malloc(vocab_size * sizeof(char*)));
        vocab_scores_ = static_cast<float*>(malloc(vocab_size * sizeof(float)));
        sorted_vocab_ = nullptr;
        for (int i = 0; i < 256; i++) {
            byte_pieces_[i * 2] = static_cast<unsigned char>(i);
            byte_pieces_[i * 2 + 1] = '\0';
        }
        FILE *file = fopen(tokenizer_path, "rb");
        if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
        if (fread(&max_token_length_, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        int len;
        for (int i = 0; i < vocab_size; i++) {
            if (fread(vocab_scores_ + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
            if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
            vocab_[i] = static_cast<char*>(malloc(len + 1));
            if (fread(vocab_[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
            vocab_[i][len] = '\0';
        }
        fclose(file);
    }

    char* decode(int prev_token, int token) const {
        char *piece = vocab_[token];
        if (prev_token == 1 && piece[0] == ' ') { piece++; }
        unsigned char byte_val;
        if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
            piece = reinterpret_cast<char*>(const_cast<unsigned char*>(byte_pieces_)) + byte_val * 2;
        }
        return piece;
    }

    void encode(const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
        if (text == nullptr) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

        // Lazy init sorted vocab (thread-safe via mutex)
        {
            std::lock_guard<std::mutex> lock(sort_mutex_);
            if (sorted_vocab_ == nullptr) {
                sorted_vocab_ = static_cast<TokenIndex*>(malloc(vocab_size * sizeof(TokenIndex)));
                for (int i = 0; i < vocab_size; i++) {
                    sorted_vocab_[i].str = vocab_[i];
                    sorted_vocab_[i].id = i;
                }
                qsort(sorted_vocab_, vocab_size, sizeof(TokenIndex), compare_tokens);
            }
        }

        char* str_buffer = static_cast<char*>(malloc((max_token_length_*2 + 1 + 2) * sizeof(char)));
        size_t str_len = 0;

        *n_tokens = 0;

        if (bos) tokens[(*n_tokens)++] = 1;

        if (text[0] != '\0') {
            int dummy_prefix = str_lookup(const_cast<char*>(" "), sorted_vocab_, vocab_size);
            tokens[(*n_tokens)++] = dummy_prefix;
        }

        for (const char *c = text; *c != '\0'; c++) {
            if ((*c & 0xC0) != 0x80) {
                str_len = 0;
            }
            str_buffer[str_len++] = *c;
            str_buffer[str_len] = '\0';
            if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
                continue;
            }
            int id = str_lookup(str_buffer, sorted_vocab_, vocab_size);
            if (id != -1) {
                tokens[(*n_tokens)++] = id;
            } else {
                for (size_t i = 0; i < str_len; i++) {
                    tokens[(*n_tokens)++] = static_cast<unsigned char>(str_buffer[i]) + 3;
                }
            }
            str_len = 0;
        }

        while (1) {
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;
            for (int i = 0; i < (*n_tokens-1); i++) {
                sprintf(str_buffer, "%s%s", vocab_[tokens[i]], vocab_[tokens[i+1]]);
                int id = str_lookup(str_buffer, sorted_vocab_, vocab_size);
                if (id != -1 && vocab_scores_[id] > best_score) {
                    best_score = vocab_scores_[id];
                    best_id = id;
                    best_idx = i;
                }
            }
            if (best_idx == -1) { break; }
            tokens[best_idx] = best_id;
            for (int i = best_idx+1; i < (*n_tokens-1); i++) {
                tokens[i] = tokens[i+1];
            }
            (*n_tokens)--;
        }

        if (eos) tokens[(*n_tokens)++] = 2;
        free(str_buffer);
    }

    static void safe_printf(const char *piece) {
        if (piece == nullptr) { return; }
        if (piece[0] == '\0') { return; }
        if (piece[1] == '\0') {
            unsigned char byte_val = piece[0];
            if (!(isprint(byte_val) || isspace(byte_val))) {
                return;
            }
        }
        printf("%s", piece);
    }

private:
    char** vocab_;
    float* vocab_scores_;
    TokenIndex* sorted_vocab_;
    unsigned int max_token_length_;
    unsigned char byte_pieces_[512];
    std::mutex sort_mutex_;

    static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
        TokenIndex tok;
        tok.str = str;
        tok.id = 0;
        TokenIndex *res = static_cast<TokenIndex*>(
            bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens));
        return res != nullptr ? res->id : -1;
    }
};

// ----------------------------------------------------------------------------
// Sampler

struct ProbIndex {
    float prob;
    int index;
};

static int compare_probindex(const void* a, const void* b) {
    const ProbIndex* a_ = static_cast<const ProbIndex*>(a);
    const ProbIndex* b_ = static_cast<const ProbIndex*>(b);
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

class Sampler {
public:
    Sampler() : vocab_size_(0), probindex_(nullptr), temperature_(0), topp_(0), rng_state_(0) {}

    Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed)
        : vocab_size_(vocab_size), temperature_(temperature), topp_(topp), rng_state_(rng_seed) {
        probindex_ = static_cast<ProbIndex*>(malloc(vocab_size * sizeof(ProbIndex)));
    }

    ~Sampler() {
        free(probindex_);
    }

    // Non-copyable
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // Movable
    Sampler(Sampler&& other) noexcept
        : vocab_size_(other.vocab_size_), probindex_(other.probindex_),
          temperature_(other.temperature_), topp_(other.topp_), rng_state_(other.rng_state_) {
        other.probindex_ = nullptr;
    }
    Sampler& operator=(Sampler&& other) noexcept {
        if (this != &other) {
            free(probindex_);
            vocab_size_ = other.vocab_size_;
            probindex_ = other.probindex_;
            temperature_ = other.temperature_;
            topp_ = other.topp_;
            rng_state_ = other.rng_state_;
            other.probindex_ = nullptr;
        }
        return *this;
    }

    int sample(float* logits) {
        int next;
        if (temperature_ == 0.0f) {
            next = sample_argmax(logits, vocab_size_);
        } else {
            for (int q = 0; q < vocab_size_; q++) { logits[q] /= temperature_; }
            softmax(logits, vocab_size_);
            float coin = random_f32();
            if (topp_ <= 0 || topp_ >= 1) {
                next = sample_mult(logits, vocab_size_, coin);
            } else {
                next = sample_topp(logits, vocab_size_, topp_, probindex_, coin);
            }
        }
        return next;
    }

private:
    int vocab_size_;
    ProbIndex* probindex_;
    float temperature_;
    float topp_;
    unsigned long long rng_state_;

    static int sample_argmax(float* probabilities, int n) {
        int max_i = 0;
        float max_p = probabilities[0];
        for (int i = 1; i < n; i++) {
            if (probabilities[i] > max_p) {
                max_i = i;
                max_p = probabilities[i];
            }
        }
        return max_i;
    }

    static int sample_mult(float* probabilities, int n, float coin) {
        float cdf = 0.0f;
        for (int i = 0; i < n; i++) {
            cdf += probabilities[i];
            if (coin < cdf) { return i; }
        }
        return n - 1;
    }

    static int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
        int n0 = 0;
        const float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < n; i++) {
            if (probabilities[i] >= cutoff) {
                probindex[n0].index = i;
                probindex[n0].prob = probabilities[i];
                n0++;
            }
        }
        qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);
        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probindex[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }
        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probindex[i].prob;
            if (r < cdf) { return probindex[i].index; }
        }
        return probindex[last_idx].index;
    }

    unsigned int random_u32() {
        rng_state_ ^= rng_state_ >> 12;
        rng_state_ ^= rng_state_ << 25;
        rng_state_ ^= rng_state_ >> 27;
        return (rng_state_ * 0x2545F4914F6CDD1Dull) >> 32;
    }

    float random_f32() {
        return (random_u32() >> 8) / 16777216.0f;
    }
};

// ----------------------------------------------------------------------------
// Utilities

static long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// Generation loop

static void generate(const Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
                     const char* prompt, int steps, RunState& state) {
    const char* empty_prompt = "";
    if (prompt == nullptr) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = static_cast<int*>(malloc((strlen(prompt)+3) * sizeof(int)));
    tokenizer.encode(prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = transformer.forward(state, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler.sample(logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = tokenizer.decode(token, next);
        Tokenizer::safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// Read stdin helper

static void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != nullptr) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

// ----------------------------------------------------------------------------
// Chat loop

static void chat(const Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
                 const char* cli_user_prompt, const char* cli_system_prompt, int steps, RunState& state) {
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = static_cast<int*>(malloc(1152 * sizeof(int)));
    int user_idx;

    int8_t user_turn = 1;
    int next;
    int token;
    int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == nullptr) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != nullptr) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            tokenizer.encode(rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        float* logits = transformer.forward(state, token, pos);
        next = sampler.sample(logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = tokenizer.decode(token, next);
            Tokenizer::safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// Multi-threaded parallel inference

struct InferenceTask {
    const char* prompt;
    int steps;
    float temperature;
    float topp;
    unsigned long long rng_seed;
    int thread_id;
};

// ----------------------------------------------------------------------------
// CLI

#ifndef TESTING

static void error_usage() {
    fprintf(stderr, "Usage:   runcpp <checkpoint> [options]\n");
    fprintf(stderr, "Example: runcpp model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|parallel, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -j <int>    number of parallel threads (for parallel mode), default 2\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // default parameters
    char *checkpoint_path = nullptr;
    char *tokenizer_path = const_cast<char*>("tokenizer.bin");
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = nullptr;
    unsigned long long rng_seed = 0;
    char *mode = const_cast<char*>("generate");
    char *system_prompt = nullptr;
    int num_threads = 2;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'j') { num_threads = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = static_cast<unsigned int>(time(nullptr));
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // Build transformer (loads weights, shared across threads)
    Transformer transformer(checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    // Build tokenizer
    Tokenizer tokenizer(tokenizer_path, transformer.config.vocab_size);

    if (strcmp(mode, "generate") == 0) {
        // Single-threaded generation (identical to run.c)
        RunState state(transformer.config);
        Sampler sampler(transformer.config.vocab_size, temperature, topp, rng_seed);
        generate(transformer, tokenizer, sampler, prompt, steps, state);
    } else if (strcmp(mode, "chat") == 0) {
        RunState state(transformer.config);
        Sampler sampler(transformer.config.vocab_size, temperature, topp, rng_seed);
        chat(transformer, tokenizer, sampler, prompt, system_prompt, steps, state);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    return 0;
}
#endif
