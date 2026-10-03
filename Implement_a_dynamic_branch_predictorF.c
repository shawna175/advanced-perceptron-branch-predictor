#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include<time.h>
#include<inttypes.h>
#include<errno.h>
#include<ctype.h>
#include<stdbool.h>

#ifndef MAX_HISTORY
#define MAX_HISTORY 63
#endif


static const size_t DEFAULT_TABLE_SIZE  = 4096;
static const int    DEFAULT_HISTORY     = 32;
static const int    DEFAULT_WEIGHT_BITS = 8;
static const size_t DEFAULT_SIM_STEPS   = 200000;
static const char  *DEFAULT_PATTERN     = "mixed";
static const size_t DEFAULT_INTERVAL    = 10000;

static int ci_cmp(const char *a, const char *b){
    while (*a && *b){
        int da = tolower((unsigned char)*a);
        int db = tolower((unsigned char)*b);
        if (da != db) return da - db;
        ++a; ++b;
    }
    return (int)((unsigned char)*a) - (int)((unsigned char)*b);
}

static unsigned popcountll_portable(uint64_t x){
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_popcountll(x);
#else
    unsigned c = 0;
    while (x) { x &= (x - 1); ++c; }
    return c;
#endif
}

static int read_line_trim(char *buf, size_t cap, FILE *in){
    if (!fgets(buf, (int)cap, in)) return 0;
    size_t n = strlen(buf);
    while (n && (buf[n-1] == '\n' || buf[n-1] == '\r')) { buf[--n] = '\0';
    }
    return 1;
}

static int is_power_of_two(size_t x){
     return x && ((x & (x-1)) == 0);
}

static int parse_size_t(const char *s, size_t *out){
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 0);
    if (errno || end == s) return 0;
    while (*end && isspace((unsigned char)*end)) ++end;
    if (*end) return 0;
    *out = (size_t)v;
    return 1;
}

static int parse_int_strict(const char *s, int *out){
    if (!s || !*s) return 0;
    errno = 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (errno || end == s) return 0;
    while (*end && isspace((unsigned char)*end)) ++end;
    if (*end) return 0;
    *out = (int)v;
    return 1;
}

static int parse_yesno(const char *s){
    if (!s || !*s) return -1;
    if (ci_cmp(s, "y") == 0 || ci_cmp(s, "yes") == 0) return 1;
    if (ci_cmp(s, "n") == 0 || ci_cmp(s, "no") == 0)  return 0;
    return -1;
}

typedef struct{
    int16_t *weights;
}
 Perceptron;

typedef struct{
    Perceptron *table;
    size_t table_size;
    int history_length;
    int weight_bits;
    int32_t weight_max;
    int32_t weight_min;
    int32_t theta;
    uint64_t ghr;
    size_t weights_per;
}
 Predictor;

static inline int32_t sat_add(int32_t cur, int32_t delta, int32_t minv, int32_t maxv){
    int64_t tmp = (int64_t)cur + (int64_t)delta;
    if (tmp > maxv) return (int32_t)maxv;
    if (tmp < minv) return (int32_t)minv;
    return (int32_t)tmp;
}

static inline uint32_t pc_hash_uint64(uint64_t pc){

    pc ^= pc >> 30; pc *= 0xbf58476d1ce4e5b9ULL;
    pc ^= pc >> 27; pc *= 0x94d049bb133111ebULL;
    pc ^= pc >> 31;
    return (uint32_t)pc;
}

Predictor *predictor_create(size_t table_size, int history_length, int weight_bits){
    if (history_length <= 0 || history_length > MAX_HISTORY){
        fprintf(stderr, "history_length must be 1..%d\n", MAX_HISTORY);
        return NULL;
    }
    if (weight_bits < 3 || weight_bits > 16){
        fprintf(stderr, "weight_bits must be 3..16\n");
        return NULL;
    }
    Predictor *P = (Predictor*)calloc(1, sizeof(Predictor));
    if (!P) return NULL;
    P->table_size = table_size;
    P->history_length = history_length;
    P->weight_bits = weight_bits;
    P->weights_per = (size_t)history_length + 1;

    int32_t vmax = (1 << (weight_bits - 1)) - 1;
    int32_t vmin = - (1 << (weight_bits - 1));
    P->weight_max = vmax; P->weight_min = vmin;

    P->theta = (193 * history_length) / 100 + 14;

    P->ghr = 0;

    P->table = (Perceptron*)calloc(table_size, sizeof(Perceptron));
    if (!P->table){
            free(P); return NULL;
    }

    for (size_t i = 0; i < table_size; ++i){
        P->table[i].weights = (int16_t*)calloc(P->weights_per, sizeof(int16_t));
        if (!P->table[i].weights){
            for (size_t j = 0; j < i; ++j) free(P->table[j].weights);
            free(P->table); free(P); return NULL;
        }
        for (int w = 0; w < (int)P->weights_per; ++w){
            int r = (rand() % 5) - 2;
            if (r > vmax) r = vmax; if (r < vmin) r = vmin;
            P->table[i].weights[w] = (int16_t)r;
        }
    }
    return P;
}

void predictor_destroy(Predictor *P){
    if (!P) return;
    for (size_t i = 0; i < P->table_size; ++i) free(P->table[i].weights);
    free(P->table);
    free(P);
}

static inline Perceptron *predictor_lookup(Predictor *P, uint64_t pc){
    uint32_t h = pc_hash_uint64(pc);
    size_t idx = h & (P->table_size - 1);
    return &P->table[idx];
}

static inline int8_t history_bit_to_signed(uint64_t ghr, int idx){
    uint64_t bit = (ghr >> idx) & 1ULL;
    return bit ? (int8_t)1 : (int8_t)-1;
}

int perceptron_predict_raw(Predictor *P, Perceptron *pcptr, int *out_sum){
    int32_t sum = (int32_t)pcptr->weights[0];
    for (int i = 0; i < P->history_length; ++i){
        int8_t  h = history_bit_to_signed(P->ghr, i);
        int16_t w = pcptr->weights[i + 1];
        sum += (int32_t)w * (int32_t)h;
    }
    if (out_sum) *out_sum = sum;
    return (sum >= 0) ? 1 : -1;
}

void perceptron_train(Predictor *P, Perceptron *pcptr, int actual){
    int32_t bias = (int32_t)pcptr->weights[0];
    bias = sat_add(bias, actual, P->weight_min, P->weight_max);
    pcptr->weights[0] = (int16_t)bias;
    for (int i = 0; i < P->history_length; ++i){
        int8_t h = history_bit_to_signed(P->ghr, i);
        int32_t cur = (int32_t)pcptr->weights[i + 1];
        int32_t delta = actual * h;
        cur = sat_add(cur, delta, P->weight_min, P->weight_max);
        pcptr->weights[i + 1] = (int16_t)cur;
    }
}

static inline void update_ghr(Predictor *P, int outcome){
    uint64_t bit = (outcome == 1) ? 1ULL : 0ULL;

    uint64_t mask = (P->history_length == 64) ? ~0ULL : ((1ULL << P->history_length) - 1ULL);
    P->ghr = ((P->ghr << 1) | bit) & mask;
}

typedef enum{
    PATTERN_LOOP,
    PATTERN_ALTERNATING,
    PATTERN_XOR,
    PATTERN_CORRELATED,
    PATTERN_RANDOM,
    PATTERN_MIXED,
    PATTERN_UNKNOWN
}
 pattern_t;

static pattern_t parse_pattern(const char *s){
    if (!s) return PATTERN_MIXED;
    if (ci_cmp(s, "loop") == 0) return PATTERN_LOOP;
    if (ci_cmp(s, "alternating") == 0 || ci_cmp(s, "alt") == 0) return PATTERN_ALTERNATING;
    if (ci_cmp(s, "xor") == 0) return PATTERN_XOR;
    if (ci_cmp(s, "correlated") == 0 || ci_cmp(s, "corr") == 0) return PATTERN_CORRELATED;
    if (ci_cmp(s, "random") == 0 || ci_cmp(s, "rand") == 0) return PATTERN_RANDOM;
    if (ci_cmp(s, "mixed") == 0) return PATTERN_MIXED;
    return PATTERN_UNKNOWN;
}

typedef struct{
    uint64_t step;
    uint64_t looplen;
    uint64_t loopcount;
    uint64_t last_pc;
    pattern_t pattern;
    uint64_t seed;
}
TraceState;

static void trace_init(TraceState *T, pattern_t p, uint64_t seed){
    T->step = 0;
    T->pattern = p;
    T->looplen = 10 + (seed % 50);
    T->loopcount = 0;
    T->last_pc = 0x1000;
    T->seed = seed;
}

static inline uint64_t simple_pc_generator(uint64_t step){
    return 0x400000ULL + (step * 16ULL) + ((step & 7ULL) << 3);
}

static void trace_next(TraceState *T, Predictor *P, uint64_t *out_pc, int *out_actual){
    uint64_t pc = simple_pc_generator(T->step);
    int actual = 1;
    switch (T->pattern){
        case PATTERN_LOOP:
            actual = ((T->step % T->looplen) == (T->looplen - 1)) ? -1 : 1;
            break;
        case PATTERN_ALTERNATING:
            actual = (T->step % 2ULL == 0ULL) ? 1 : -1;
            pc ^= ((T->step >> 1) & 0xFFULL);
            break;
        case PATTERN_XOR:{
            uint64_t mix = P->ghr ^ pc;
            unsigned ones = popcountll_portable(mix);
            actual = (ones % 2U == 0U) ? 1 : -1;
            break;
        }
        case PATTERN_CORRELATED:{
            int idx = (int)((T->step / 7ULL) % (uint64_t)(P->history_length == 0 ? 1 : P->history_length));
            int8_t h = history_bit_to_signed(P->ghr, idx);
            actual = (h == 1) ? 1 : -1;
            if ((T->step & 31ULL) == 0ULL) actual = -actual;
            break;
        }
        case PATTERN_RANDOM:
            actual = (rand() & 1) ? 1 : -1;
            break;
        case PATTERN_MIXED:{
            uint64_t block = (T->step / 500ULL);
            switch (block % 4ULL) {
                case 0ULL: actual = ((T->step % (10ULL + ((T->seed + block) % 40ULL))) == 9ULL) ? -1 : 1; break;
                case 1ULL: actual = (T->step % 2ULL == 0ULL) ? 1 : -1; break;
                case 2ULL: actual = (popcountll_portable(P->ghr ^ pc) % 2U == 0U) ? 1 : -1; break;
                default:   actual = (rand() & 1) ? 1 : -1; break;
            }
            break;
        }
        default:
            actual = (rand() & 1) ? 1 : -1;
            break;
    }
    pc ^= (uint64_t)((T->step * 0x9e3779b9ULL) & 0xFFFULL);
    *out_pc = pc;
    *out_actual = actual;
    T->step++;
}

static int read_trace_file_step(FILE *f, uint64_t *out_pc, int *out_actual){
    char line[256];
    while (fgets(line, (int)sizeof(line), f)){
        char *s = line;
        while (*s && isspace((unsigned char)*s)) ++s;
        if (*s == '\0' || *s == '#' || *s == '\n') continue;
        errno = 0;
        char *end = NULL;
        unsigned long long pc_ull = strtoull(s, &end, 0);
        if (errno || end == s) continue;
        uint64_t pc = (uint64_t)pc_ull;
        while (*end && isspace((unsigned char)*end)) ++end;
        int t = 1;
        if (*end == '0' || *end == '1'){
            t = (*end == '0') ? 0 : 1;
        }
         else{
            int tk = 0;
            if (sscanf(end, "%d", &tk) == 1) t = (tk ? 1 : 0);
        }
        *out_pc = pc;
        *out_actual = (t ? 1 : -1);
        return 1;
    }
    return 0;
}

typedef struct{
    uint64_t total;
    uint64_t correct;
    uint64_t trained;
    uint64_t low_conf_updates;
}
Stats;

static void predictor_step(Predictor *P, uint64_t pc, int actual, Stats *S){
    Perceptron *pcptr = predictor_lookup(P, pc);
    int sum = 0;
    int pred = perceptron_predict_raw(P, pcptr, &sum);
    if (pred == actual) S->correct++;
    S->total++;
    if (pred != actual || abs(sum) <= P->theta){
        perceptron_train(P, pcptr, actual);
        S->trained++;
        if (abs(sum) <= P->theta) S->low_conf_updates++;
    }
    update_ghr(P, actual);
}

static void print_header(void){
    printf("=============================================================\n");
    printf("  Advanced Perceptron Branch Predictor (Interactive)\n");
    printf("=============================================================\n");
}

static void prompt_string(const char *label, const char *def, char *out, size_t cap){
    printf("%s [%s]: ", label, def);
    if (!read_line_trim(out, cap, stdin)){
            out[0] = '\0';
    }
    if (out[0] == '\0') snprintf(out, cap, "%s", def);
}

static size_t prompt_table_size(void){
    char buf[64];
    for (;;){
        printf("Perceptron table size (power of two) [%zu]: ", DEFAULT_TABLE_SIZE);
        if (!read_line_trim(buf, sizeof(buf), stdin)) { return DEFAULT_TABLE_SIZE;
         }
        if (buf[0] == '\0') return DEFAULT_TABLE_SIZE;
        size_t v = 0;
        if (parse_size_t(buf, &v) && is_power_of_two(v)) return v;
        printf("  -> Please enter a power of two (e.g., 1024, 2048, 4096).\n");
    }
}

static int prompt_history_length(void){
    char buf[64];
    for (;;){
        printf("History length m [default %d, max %d]: ", DEFAULT_HISTORY, MAX_HISTORY);
        if (!read_line_trim(buf, sizeof(buf), stdin)){
                return DEFAULT_HISTORY;
        }
        if (buf[0] == '\0') return DEFAULT_HISTORY;
        int v = 0;
        if (parse_int_strict(buf, &v) && v >= 1 && v <= MAX_HISTORY) return v;
        printf("  -> Enter an integer in 1..%d.\n", MAX_HISTORY);
    }
}

static int prompt_weight_bits(void){
    char buf[64];
    for (;;){
        printf("Weight bits (signed) [3..16, default %d]: ", DEFAULT_WEIGHT_BITS);
        if (!read_line_trim(buf, sizeof(buf), stdin)){
                return DEFAULT_WEIGHT_BITS;
        }
        if (buf[0] == '\0') return DEFAULT_WEIGHT_BITS;
        int v = 0;
        if (parse_int_strict(buf, &v) && v >= 3 && v <= 16) return v;
        printf("  -> Enter an integer in 3..16.\n");
    }
}

static size_t prompt_sim_steps(void){
    char buf[64];
    for (;;){
        printf("Simulation steps [%zu]: ", DEFAULT_SIM_STEPS);
        if (!read_line_trim(buf, sizeof(buf), stdin)){
                return DEFAULT_SIM_STEPS;
        }
        if (buf[0] == '\0') return DEFAULT_SIM_STEPS;
        size_t v = 0;
        if (parse_size_t(buf, &v) && v > 0) return v;
        printf("  -> Enter a positive integer.\n");
    }
}

static size_t prompt_interval(void){
    char buf[64];
    for (;;) {
        printf("Interval for summary prints [%zu]: ", DEFAULT_INTERVAL);
        if (!read_line_trim(buf, sizeof(buf), stdin)){
                return DEFAULT_INTERVAL;
         }
        if (buf[0] == '\0') return DEFAULT_INTERVAL;
        size_t v = 0;
        if (parse_size_t(buf, &v) && v > 0) return v;
        printf("  -> Enter a positive integer.\n");
    }
}

static int prompt_use_trace(char *path_out, size_t cap){
    char yn[16];
    for (;;){
        printf("Read from trace file instead of synthetic pattern? (y/n) [n]: ");
        if (!read_line_trim(yn, sizeof(yn), stdin)) return 0;
        if (yn[0] == '\0') return 0;
        int v = parse_yesno(yn);
        if (v == 1){
            prompt_string("Trace file path", "", path_out, cap);
            return 1;
        }
        if (v == 0) return 0;
        printf("  -> Please type y or n.\n");
    }
}

static int prompt_pattern(char *pattern_out, size_t cap, int *out_enum_value){
    printf("Pattern options: loop | alternating | xor | correlated | random | mixed\n");
    prompt_string("Choose pattern", DEFAULT_PATTERN, pattern_out, cap);
    pattern_t p = parse_pattern(pattern_out);
    if (p == PATTERN_UNKNOWN){
        printf("  -> Unknown pattern. Using default '%s'.\n", DEFAULT_PATTERN);
        snprintf(pattern_out, cap, "%s", DEFAULT_PATTERN);
        p = parse_pattern(pattern_out);
    }
    *out_enum_value = (int)p;
    return 1;
}

int main(void){
    print_header();

    size_t table_size = prompt_table_size();
    int    history_length = prompt_history_length();
    int    weight_bits = prompt_weight_bits();
    size_t sim_steps = prompt_sim_steps();
    size_t interval = prompt_interval();

    char trace_path[256]; trace_path[0] = '\0';
    int use_trace = prompt_use_trace(trace_path, sizeof(trace_path));

    char pattern_buf[32]; pattern_buf[0] = '\0';
    int p_enum = (int)PATTERN_MIXED;
    if (!use_trace){
        prompt_pattern(pattern_buf, sizeof(pattern_buf), &p_enum);
    }
     else{
        printf("Trace mode selected. Pattern choice will be ignored.\n");
    }

    if (!is_power_of_two(table_size)){
        fprintf(stderr, "Error: table size must be a power of two.\n");
        return 1;
    }
    if (history_length < 1 || history_length > MAX_HISTORY){
        fprintf(stderr, "Error: history length must be 1..%d\n", MAX_HISTORY);
        return 1;
    }
    if (weight_bits < 3 || weight_bits > 16){
        fprintf(stderr, "Error: weight bits must be 3..16\n");
        return 1;
    }

    unsigned seed = (unsigned)time(NULL) ^ (unsigned)clock();
    srand(seed);

    Predictor *P = predictor_create(table_size, history_length, weight_bits);
    if (!P){
            fprintf(stderr, "predictor_create failed\n"); return 1;
    }

    printf("\nConfiguration:\n");
    printf("  table_size      = %zu\n", table_size);
    printf("  history_length  = %d\n", history_length);
    printf("  weight_bits     = %d\n", weight_bits);
    printf("  theta           = %d\n", P->theta);
    printf("  sim_steps       = %zu\n", sim_steps);
    printf("  interval        = %zu\n", interval);
    if (use_trace){
        printf("  trace_file      = %s\n", trace_path[0] ? trace_path : "(empty path)");
    }
     else{
        printf("  pattern         = %s\n", pattern_buf[0] ? pattern_buf : DEFAULT_PATTERN);
    }
    printf("\n");

    Stats S = (Stats){0};
    TraceState T;
    trace_init(&T, (pattern_t)p_enum, (uint64_t)rand());

    FILE *tf = NULL;
    if (use_trace && trace_path[0]){
        tf = fopen(trace_path, "r");
        if (!tf){
            perror("fopen trace file");
            predictor_destroy(P);
            return 1;
        }
    }

    size_t step = 0;
    clock_t c0 = clock();
    while (step < sim_steps){
        uint64_t pc;
        int actual;
        if (tf){
            int ok = read_trace_file_step(tf, &pc, &actual);
            if (!ok) break;
        }
         else{
            trace_next(&T, P, &pc, &actual);
        }

        predictor_step(P, pc, actual, &S);

        if (interval > 0 && step > 0 && (step % interval) == 0){
            double acc = (S.total == 0) ? 0.0 : (100.0 * (double)S.correct / (double)S.total);
            printf("Step %zu  total=%" PRIu64 "  acc=%.3f%%  trained=%" PRIu64 "  lowconf=%" PRIu64 "\n",
                   step, S.total, acc, S.trained, S.low_conf_updates);
        }
        step++;
    }
    clock_t c1 = clock();
    double elapsed = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;

    printf("\n==== Final Results ====\n");
    printf("Simulated steps: %zu\n", step);
    printf("Total predictions: %" PRIu64 "\n", S.total);
    printf("Correct predictions: %" PRIu64 "\n", S.correct);
    printf("Mispredictions: %" PRIu64 "\n", S.total - S.correct);
    double acc = (S.total == 0) ? 0.0 : (100.0 * (double)S.correct / (double)S.total);
    printf("Accuracy: %.4f%%\n", acc);
    printf("Trained updates: %" PRIu64 "  (updates when wrong OR low confidence)\n", S.trained);
    printf("Low-confidence updates (|sum| <= theta): %" PRIu64 "\n", S.low_conf_updates);
    printf("Weight range: [%d, %d]\n", P->weight_min, P->weight_max);
    printf("Theta (training threshold): %d\n", P->theta);
    printf("Elapsed CPU time: %.3fs\n", elapsed);
    printf("Predictions/sec (approx): %.0f\n", (double)S.total / (elapsed > 0 ? elapsed : 1e-6));

    if (tf) fclose(tf);
    predictor_destroy(P);
    return 0;
}
