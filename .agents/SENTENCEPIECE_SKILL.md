# SentencePiece Unigram Tokenizer Runtime - Implementation Guide

This document captures the critical implementation knowledge needed for a clean-room Rust reimplementation of the SentencePiece Unigram tokenizer runtime. It focuses on the inference/runtime path, not training.

**Note**: This is NOT a translation of C++ to Rust. It's a practical engineering guide for understanding the algorithms, data structures, and behaviors that must be preserved.

---

## High-Level Architecture

The SentencePiece runtime consists of these key components:

```
Input Text
    ↓
SentencePieceProcessor
    ↓
Normalizer (text normalization)
    ↓
ModelInterface (UnigramModel)
    ↓
Lattice Construction (token candidates)
    ↓
Viterbi (best path finding)
    ↓
SentencePieceText (output with metadata)
```

### Key Classes and Their Responsibilities

| Class                    | File                           | Responsibility                                           |
| ------------------------ | ------------------------------ | -------------------------------------------------------- |
| `SentencePieceProcessor` | `sentencepiece_processor.cc/h` | Main API, orchestrates normalization + encoding/decoding |
| `UnigramModel`           | `unigram_model.cc/h`           | Unigram tokenization algorithm, Viterbi search           |
| `Lattice`                | `unigram_model.cc/h`           | Search space representation, node-based graph            |
| `Normalizer`             | `normalizer.cc/h`              | Text normalization with precompiled rules                |
| `PrefixMatcher`          | `normalizer.cc/h`              | Fast longest-prefix matching for user-defined symbols    |
| `ModelInterface`         | `model_interface.cc/h`         | Base class for all model types                           |
| `ModelProto`             | `sentencepiece_model.proto`    | Protobuf model format                                    |

---

## Important Source Files

### Core Runtime Files

- `src/unigram_model.cc` - Unigram model implementation, Viterbi, NBest, sampling
- `src/unigram_model.h` - Unigram model interface, Lattice class
- `src/sentencepiece_processor.cc` - Main processor, encode/decode pipelines
- `src/sentencepiece_processor.h` - Public API definitions
- `src/normalizer.cc` - Text normalization implementation
- `src/normalizer.h` - Normalizer interface
- `src/model_interface.cc` - Base model class
- `src/model_interface.h` - Model interface definitions
- `src/util.cc` - Utility functions
- `src/util.h` - UTF-8 handling, string utilities

### Protocol Buffer Definitions

- `src/sentencepiece_model.proto` - Model format definition
  - `ModelProto` - Top-level model container
  - `TrainerSpec` - Training parameters (some needed at runtime)
  - `NormalizerSpec` - Normalization configuration
  - `ModelProto.SentencePiece` - Individual token definition

### Test Files (Golden Reference)

- `src/unigram_model_test.cc` - Unigram model tests
- `src/sentencepiece_processor_test.cc` - Processor tests
- `src/normalizer_test.cc` - Normalizer tests

---

## Model Format (`.model` files)

SentencePiece models are serialized Protocol Buffers. Key fields for runtime:

### ModelProto Structure

```protobuf
message ModelProto {
  message SentencePiece {
    enum Type {
      NORMAL = 1;        // Regular token
      UNKNOWN = 2;       // <unk> token
      CONTROL = 3;       // <s>, </s>, custom control symbols
      USER_DEFINED = 4;  // User-defined symbols (always kept as single token)
      BYTE = 6;          // Byte fallback tokens (<0x00> to <0xFF>)
      UNUSED = 5;        // Disabled token (skipped during encoding)
    }
    string piece = 1;      // Token string (e.g., "▁hello")
    float score = 2;       // Log probability
    Type type = 3;         // Token type
  }
  repeated SentencePiece pieces = 1;           // Vocabulary
  TrainerSpec trainer_spec = 2;               // Training config
  NormalizerSpec normalizer_spec = 3;         // Normalization rules
  NormalizerSpec denormalizer_spec = 5;       // Denormalization rules
  SelfTestData self_test_data = 4;           // Self-test samples
}
```

### TrainerSpec Fields Used at Runtime

- `unk_id`, `bos_id`, `eos_id`, `pad_id` - Special token IDs
- `unk_piece`, `bos_piece`, `eos_piece`, `pad_piece` - Special token strings
- `unk_surface` - What <unk> renders as (default: "   " U+2047)
- `byte_fallback` - Enable byte fallback for unknown characters
- `control_symbols` - List of control symbols
- `user_defined_symbols` - List of user-defined symbols

### NormalizerSpec Fields

- `precompiled_charsmap` - Compiled normalization rules (binary trie + normalized strings)
- `add_dummy_prefix` - Add dummy whitespace at beginning (default: true)
- `remove_extra_whitespaces` - Remove leading/trailing/duplicate whitespace (default: true)
- `escape_whitespaces` - Replace whitespace with meta symbol (default: true)

---

## Vocabulary Representation

### Token Types and Their Behavior

1. **NORMAL (Type 1)** - Regular vocabulary tokens
   - Have a log probability score
   - Used in Viterbi search

2. **UNKNOWN (Type 2)** - The `<unk>` token
   - Only one unknown token exists (usually at ID 0)
   - Used when no vocabulary piece matches

3. **CONTROL (Type 3)** - Special control symbols
   - `<s>` (BOS), `</s>` (EOS), custom control symbols
   - Invisible in decoded text (empty surface)
   - Always have ID based on `trainer_spec.bos_id`, `eos_id`

4. **USER_DEFINED (Type 4)** - User-defined symbols
   - Always treated as a single token
   - Get a very high score bonus to ensure selection
   - Common usage: placeholders for named entities

5. **BYTE (Type 6)** - Byte fallback tokens
   - Format: `<0x00>` to `<0xFF>`
   - Used when `byte_fallback` is enabled
   - Each byte of unknown character gets its own token

6. **UNUSED (Type 5)** - Disabled tokens
   - Skipped during lattice population
   - Not available for encoding

### Token ID Layout

```
ID 0: <unk> (UNKNOWN)
ID 1: <s> (CONTROL)
ID 2: </s> (CONTROL)
ID 3+: Regular tokens (NORMAL, USER_DEFINED, BYTE, UNUSED)
```

Note: Control symbols are added first, so their IDs are fixed and known.

### Score Values

- Scores are **log probabilities** (typically negative)
- Higher score = more likely token
- User-defined symbols get bonus score to ensure selection
- Unknown token gets penalty score: `min_score - kUnkPenalty` (where `kUnkPenalty = 10.0`)

---

## Normalization Pipeline

### Normalization Steps (in order)

The normalizer performs these transformations on input text:

1. **Remove leading whitespace** (if `remove_extra_whitespaces`)
2. **Add dummy prefix** (if `add_dummy_prefix`)
   - Adds U+2581 (LOWER ONE EIGHT BLOCK) at the beginning
   - This ensures "hello" and "hello world" are tokenized similarly
3. **Character normalization** (using precompiled charsmap)
   - NFKC normalization
   - Full-width to half-width conversion
   - User-defined character mappings
   - Uses leftmost-longest matching with a trie
4. **Replace spaces** (if `escape_whitespaces`)
   - Regular space ' ' → U+2581 (▁)
5. **Remove extra whitespace**
   - Remove duplicate internal spaces
   - Remove trailing whitespace (if `remove_extra_whitespaces`)
6. **Add dummy suffix** (if `treat_whitespace_as_suffix` and `add_dummy_prefix`)
   - Rare option, adds whitespace at end instead

### Key Normalization Constants

```cpp
const char kSpaceSymbol[] = "\xe2\x96\x81";  // U+2581 LOWER ONE EIGHT BLOCK
const char kDefaultUnknownSymbol[] = " \xE2\x81\x87 ";  // U+2047 DOUBLE QUESTION MARK
const char kReplacementCharacter[] = "\xef\xbf\xbd";  // U+FFFD REPLACEMENT CHARACTER
```

### Normalization Algorithm (Leftmost-Longest Matching)

The normalizer uses a trie-based approach:

```cpp
for each position in input:
    1. Check user-defined symbols (PrefixMatcher)
    2. Find longest match in normalization trie
    3. If no match, consume one Unicode character
    4. Apply character-level transformations
    5. Track alignment (norm_to_orig mapping)
```

### Alignment Mapping

The normalizer returns both:

- `normalized` - the normalized string
- `norm_to_orig` - vector mapping each normalized byte to original byte position

This alignment is used later to map token positions back to the original text.

---

## Whitespace Marker Handling

### The Space Symbol (U+2581)

SentencePiece uses U+2581 (LOWER ONE EIGHT BLOCK, "▁") as a whitespace marker:

**Why?** It's a visible character that can represent spaces in token strings.

**Usage:**

- Normalization replaces ' ' with '▁' (if `escape_whitespaces`)
- Tokens typically start with '▁' to indicate word boundaries
- Example: "hello world" → "▁hello ▁world"

**Encoding behavior:**

- '▁hello' → one token (word with leading space)
- 'hello' → one token (word without leading space)

**Decoding behavior:**

- Replace '▁' with ' ' in the output
- Exception: Leading '▁' may be stripped based on `add_dummy_prefix`

**Dummy Prefix:**

- When `add_dummy_prefix=true`, the normalizer adds a leading '▁'
- This ensures consistent tokenization of "word" vs "hello word"
- During decoding, this leading '▁' may be removed

---

## Unknown Token Handling

### When Unknown Tokens Appear

Unknown tokens are used when:

1. No vocabulary piece matches at a position
2. A character is not in the vocabulary
3. Byte fallback is not enabled

### Unknown Token Properties

- ID: 0 (always, if `unk_id=0`)
- Type: UNKNOWN
- Score: `min_score - kUnkPenalty` (very low to discourage selection)
- Surface: `unk_surface` (default: "   " U+2047)

### Unknown Sequence Merging

During decoding, consecutive unknown tokens are merged:

```
Encoded: ["hello", "▁", "▁", "▁world"]
Decoded: "hello   world"
```

But if multiple consecutive unknown pieces represent the same characters:

```
Encoded: ["hello", "▁<unk>", "▁<unk>"]  // E.g., for "EF"
Decoded: "hello  EF"  // Merged to one unknown surface
```

This helps decoder copy or generate unknown tokens more easily.

---

## Byte Fallback Handling

### What is Byte Fallback?

When enabled, unknown characters are decomposed into their UTF-8 bytes, and each byte is tokenized separately.

### Byte Fallback Process

1. During encoding, if a character is unknown:
   - Convert character to UTF-8 bytes
   - Create byte tokens: `<0x00>` to `<0xFF>`
   - These are added to the vocabulary with type BYTE

2. Example: "あ" (U+3042) = 0xE3 0x81 0x82
   - Encodes to: ["<0xE3>", "<0x81>", "<0x82>"]

3. During decoding:
   - Byte tokens are reconstructed into characters
   - UTF-8 validation is performed
   - Invalid UTF-8 sequences become U+FFFD (replacement character)

### Byte Token Format

- String: `<0x00>` to `<0xFF>` (hex format)
- Type: BYTE
- Score: Usually very low

### Decoding Byte Sequences

```
Byte tokens: ["<0xE3>", "<0x81>", "<0x82>"]
Bytes: [0xE3, 0x81, 0x82]
Decoded character: "あ"

Invalid sequence: ["<0xFF>", "<0xFF>"]
Bytes: [0xFF, 0xFF]  (Invalid UTF-8)
Decoded character: U+FFFD (replacement)
```

### Surface Mapping for Byte Pieces

When decoding, only the **last byte piece** in a multi-byte sequence shows the character:

- `<0xE3>`: surface = ""
- `<0x81>`: surface = ""
- `<0x82>`: surface = "あ"

This is because the byte pieces together form one character.

---

## BOS/EOS Handling

### Special Token IDs

- `<s>` (BOS - Beginning of Sentence): ID from `trainer_spec.bos_id` (default: 1)
- `</s>` (EOS - End of Sentence): ID from `trainer_spec.eos_id` (default: 2)

### BOS/EOS in Encoding

By default, BOS/EOS are **not** automatically added during encoding. They must be explicitly added using extra options:

```cpp
// Add BOS/EOS to encoding
processor.SetEncodeExtraOptions("bos:eos");
```

### BOS/EOS in Decoding

During decoding:

- Control tokens (including BOS/EOS) have **empty surface**
- They are invisible in the decoded text
- They don't contribute to the output string

### Extra Options System

The processor supports extra options that modify encoding/decoding:

| Option      | Effect                            |
| ----------- | --------------------------------- |
| `bos`       | Add <s> at beginning              |
| `eos`       | Add </s> at end                   |
| `reverse`   | Reverse the input before encoding |
| `unk_piece` | Add <unk> token (rarely used)     |

Multiple options can be combined: `"bos:eos"`

---

## Unigram Tokenization Mechanism

### Unigram Language Model

The Unigram model assumes each piece is generated independently:

```
P(text) = P(piece1) × P(piece2) × ... × P(pieceN)
```

In log space:

```
log P(text) = log P(piece1) + log P(piece2) + ... + log P(pieceN)
```

This assumption enables efficient Viterbi decoding.

### Tokenization as a Graph Search Problem

Given text "ABC", possible tokenizations:

- "A" + "B" + "C"
- "AB" + "C"
- "A" + "BC"
- "ABC"

Each tokenization is a path through a lattice. We find the path with highest total score.

### Viterbi Algorithm

**Goal**: Find the highest-scoring sequence of tokens.

**Algorithm** (forward pass):

```cpp
for each position pos from 0 to len:
    for each node rnode that starts at pos:
        best_score = -infinity
        best_node = null
        for each node lnode that ends at pos:
            score = lnode.backtrace_score + rnode.score
            if score > best_score:
                best_score = score
                best_node = lnode
        rnode.prev = best_node
        rnode.backtrace_score = best_score
```

**Backward pass**:

```cpp
results = []
node = end_nodes[len][0]
while node.prev != null:
    results.push(node)
    node = node.prev
reverse(results)
```

**Key insight**: Due to unigram assumption, we can compute the best path ending at each position independently and store it. This enables the optimized encoder.

### Optimized Encoder (EncodeOptimized)

The optimized encoder achieves 2.1x speedup by:

1. **No full lattice construction**: Only stores best path ending at each position
2. **Direct UTF-8 work**: Works with byte positions instead of Unicode positions
3. **Minimal state**: Only 3 fields per position (vs 8 in original Node)

**Data structure**:

```cpp
struct BestPathNode {
    int id;              // Vocab ID (or -1 for UNK)
    float best_path_score;  // Best total score ending at this position
    int starts_at;       // Starting position for backtracking
}
```

**Algorithm**:

```cpp
for each starting position:
    for each matching piece (using trie.traverse):
        candidate_score = piece.score + best_path_ends_at[start].best_path_score
        if candidate_score > best_path_ends_at[end].best_path_score:
            update best_path_ends_at[end]

// Backtrack to find the path
results = []
pos = end
while pos > 0:
    node = best_path_ends_at[pos]
    results.append((text[node.starts_at:pos], node.id))
    pos = node.starts_at
reverse(results)
```

**Time complexity**: O(n × k) where n = text length, k = avg matches per position
**Space complexity**: O(n) vs O(n × k) for original

---

## Lattice Construction

### What is a Lattice?

A lattice is a directed acyclic graph representing all possible tokenizations of the text.

**Example for "ABC":**

```
      ┌── A ──┐
BOS ─┤       ├──┬── AB ──┐
      └── AB ─┘  │        ├── C ──┐
                 └── ABC ─┤        ├── EOS
                         └── BC ──┘
```

### Lattice Data Structures

```cpp
class Lattice {
    // Positions indexed by Unicode character position
    std::vector<std::vector<Node*>> begin_nodes_;  // Nodes starting at each position
    std::vector<std::vector<Node*>> end_nodes_;    // Nodes ending at each position

    // Node structure
    struct Node {
        absl::string_view piece;  // Token text
        uint32_t pos;             // Start position (Unicode)
        uint32_t length;          // Length in Unicode characters
        int id;                   // Vocab ID (or -1 for UNK)
        float score;              // Log probability
        float backtrace_score;    // Best score ending at this node
        Node* prev;               // Previous node in best path
    };
};
```

### PopulateNodes: Building the Lattice

**Algorithm**:

```cpp
for each Unicode position begin_pos in sentence:
    // Find all pieces that are prefixes of text[begin_pos:]
    results = trie.commonPrefixSearch(text[begin_pos:])

    has_single_node = false
    for each result in results:
        if result.length == 1:  // Single character
            has_single_node = true
        if piece is UNUSED: continue
        node = lattice.insert(begin_pos, result.length)
        node.id = result.id
        node.score = piece.score  // or bonus for USER_DEFINED

    // Ensure at least one node (UNK) at each position
    if not has_single_node:
        node = lattice.insert(begin_pos, 1)
        node.id = unk_id_
        node.score = min_score - kUnkPenalty
```

**Key points**:

- Uses `commonPrefixSearch` on trie to find all matches
- Each position must have at least one node (UNK)
- User-defined symbols get bonus score
- UNUSED pieces are skipped

### Trie Construction

The trie is built from vocabulary pieces:

```cpp
void BuildTrie(pieces):
    sort(pieces)  // Required by DoubleArray::build()

    for each piece in pieces:
        key[i] = piece.data()
        length[i] = piece.size()
        value[i] = piece.id

    trie.build(keys, lengths, values)

    // Compute max results for buffer allocation
    for each piece:
        num = trie.commonPrefixSearch(piece)
        trie_results_size = max(trie_results_size, num)
```

**Trie properties**:

- Maps piece strings → vocab IDs
- Double-array trie for O(k) prefix search
- `commonPrefixSearch` returns all matching prefixes

---

## Viterbi Scoring Explanation

### Score Representation

- Scores are **log probabilities** (typically negative)
- Higher score = more likely token
- Total path score = sum of individual token scores

### Scoring Example

```
Text: "ABC"
Possible paths:
  "A" + "B" + "C":    -1.0 + -1.0 + -1.0 = -3.0
  "AB" + "C":        -2.0 + -1.0 = -3.0
  "A" + "BC":        -1.0 + -3.0 = -4.0
  "ABC":             -5.0 = -5.0

Best path: "A" + "B" + "C" (score -3.0) or "AB" + "C" (score -3.0)
```

If two paths have equal score, the first one found is returned (deterministic).

### User-Defined Symbol Scoring

User-defined symbols get a large bonus score to ensure they're always selected:

```cpp
if is_user_defined:
    score = GetUserDefinedScore(length)  // Large positive value
else:
    score = piece.score
```

This effectively forces the Viterbi algorithm to always choose user-defined symbols.

### Unknown Token Scoring

Unknown tokens get a penalty:

```cpp
unk_score = min_score - kUnkPenalty  // min_score - 10.0
```

This discourages the algorithm from choosing unknown tokens unless necessary.

---

## Probability / Score Handling

### Log-Sum-Exp Trick

When computing probabilities in log space, we use the log-sum-exp trick for numerical stability:

```cpp
log_sum_exp(x, y) = log(exp(x) + exp(y))
                 = max(x, y) + log(exp(x - max) + exp(y - max))
```

This prevents overflow when exponentiating large negative numbers.

### Forward Algorithm

Computes forward probabilities for each node:

```cpp
ForwardAlgorithm(inv_theta):
    for each position:
        for each node at position:
            alpha[node] = log(sum(exp(alpha[prev] + inv_theta * node.score)))
```

### Backward Algorithm

Computes backward probabilities (for training):

```cpp
BackwardAlgorithm(inv_theta):
    for each position (reverse):
        beta[node] = log(sum(exp(beta[next] + inv_theta * next.score)))
```

### Marginal Probabilities

Used in EM algorithm during training:

```cpp
PopulateMarginal(freq, expected):
    alpha = ForwardAlgorithm(inv_theta)
    beta = BackwardAlgorithm(inv_theta)
    Z = alpha[eos]

    for each node:
        marginal_prob = exp(alpha[node] + beta[node] - Z)
        expected[node.id] += marginal_prob * freq
```

### Sampling Probabilities

For stochastic sampling, we compute probabilities based on scores:

```cpp
prob(path) = exp(inv_theta * path_score) / Z
```

Where `inv_theta` is the inverse temperature parameter:

- `inv_theta = 0.0`: Uniform sampling
- `inv_theta = 1.0`: Proportional to model probabilities
- `inv_theta > 1.0`: More confident in model

---

## Normalization Behavior

### Default Normalization Rules

The default normalizer ("nmt_nfkc") includes:

1. **NFKC compatibility decomposition**
2. **Full-width to half-width conversion**
   - Full-width ASCII → Half-width ASCII
   - Full-width punctuation → Half-width punctuation
3. **Specific character mappings**
   - ㍿ → 株式会社
   - No. → No.
   - Various other mappings

### Normalization Examples

| Input         | Normalized     |
| ------------- | -------------- |
| "Ｈｅｌｌｏ"  | "▁Hello"       |
| "㍿"          | "▁株式会社"    |
| "Hello World" | "▁Hello▁World" |
| " Hello "     | "▁Hello"       |

### Unicode Script Handling

The normalizer can split by Unicode script (if `split_by_unicode_script`):

- Prevents mixing scripts in a single token
- Exception: CJK characters (Chinese/Japanese/Korean) are treated as one script
- Example: "F1" would be split into "F" and "1" if this is enabled

### Number Handling

If `split_by_number` is enabled:

- Splits at number/non-number boundaries
- Example: "F1" → "F" + "1"

---

## Unicode / UTF-8 Considerations

### UTF-8 Length Detection

SentencePiece uses a lookup table for fast UTF-8 length detection:

```cpp
// First 4 bits of first byte determine length
"\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[byte >> 4]

// 0x00-0x7F:   1 byte
// 0x80-0xBF:   continuation byte (invalid first byte)
// 0xC0-0xDF:   2 bytes
// 0xE0-0xEF:   3 bytes
// 0xF0-0xF7:   4 bytes
```

### Valid UTF-8 Checking

```cpp
IsValidDecodeUTF8(input, mblen):
    c = DecodeUTF8(input, mblen)
    return (c != kUnicodeError) || (*mblen == 3)  // 3-byte replacement char
```

### Unicode Character Length

The lattice operates on **Unicode character positions**, but the optimized encoder works on **UTF-8 byte positions**:

**Example: "テストab"**

- Unicode length: 5 characters
- UTF-8 length: 11 bytes
- Lattice positions: 0, 1, 2, 3, 4 (Unicode)
- Optimized encoder positions: 0, 3, 6, 9, 10, 11 (bytes)

### Codepoint Validation

Valid Unicode codepoints:

- Range: U+0000 to U+10FFFF
- Exclude: U+D800 to U+DFFF (surrogate pairs, for UTF-16)
- Invalid values become U+FFFD (replacement character)

---

## Whitespace Marker Behavior (Detailed)

### Normalization Phase

**Input**: "Hello World"

1. Strip leading space (if `remove_extra_whitespaces`): "Hello World"
2. Add dummy prefix (if `add_dummy_prefix`): "▁Hello World"
3. Normalize characters: "▁Hello World"
4. Escape spaces: "▁Hello▁World"
5. Remove duplicate spaces: "▁Hello▁World"
6. Strip trailing space (if `remove_extra_whitespaces`): "▁Hello▁World"

**Output**: "▁Hello▁World"

### Encoding Phase

The encoder sees "▁Hello▁World" and tokenizes it.

**Possible tokens**:

- "▁Hello" (word with leading space)
- "▁World" (word with leading space)
- "▁" (space alone, rare)
- "Hello" (word without space, rare)
- "World" (word without space, rare)

### Decoding Phase

**Input tokens**: ["▁Hello", "▁World"]

1. Process control tokens (BOS/EOS): skip (empty surface)
2. Decode "▁Hello": Replace "▁" with " " → " Hello"
3. Decode "▁World": Replace "▁" with " " → " World"
4. Concatenate: " Hello World"
5. Strip leading space (if `add_dummy_prefix` and not `treat_whitespace_as_suffix`)

**Output**: "Hello World"

### Edge Cases

1. **Multiple spaces**: "Hello World" → "▁Hello▁World" (spaces collapsed)
2. **Leading space**: " Hello" → "▁Hello" (then stripped on decode)
3. **Trailing space**: "Hello " → "▁Hello" (stripped on decode)
4. **Space-only**: " " → "" (all spaces removed)

---

## Runtime Encode Pipeline (Step-by-Step)

### Step 1: Input Validation

```cpp
if input is empty:
    return empty result
```

### Step 2: Text Normalization

```cpp
normalized, norm_to_orig = normalizer.Normalize(input)
```

**Normalization does**:

- Character normalization (NFKC, etc.)
- Add dummy prefix (▁)
- Escape spaces ( → ▁)
- Remove extra whitespace

**norm_to_orig mapping**:

- `norm_to_orig[i]` = original byte position for normalized byte `i`
- Used to map token positions back to original text

### Step 3: Model Encoding

**Standard encoder** (with full lattice):

```cpp
lattice.SetSentence(normalized)
model.PopulateNodes(&lattice)
results = lattice.Viterbi()
```

**Optimized encoder** (default):

```cpp
results = model.EncodeOptimized(normalized)
```

### Step 4: Populate Output

```cpp
for each (piece, id) in results:
    spt.add_piece(piece, id)
    // Map positions using norm_to_orig
```

### Step 5: Apply Extra Options

```cpp
if encode_extra_options contains "bos":
    prepend <s>
if encode_extra_options contains "eos":
    append </s>
if encode_extra_options contains "reverse":
    reverse the tokens
```

### Step 6: Return Results

```cpp
return pieces as strings
return pieces as IDs
return SentencePieceText with metadata
```

---

## Runtime Decode Pipeline (Step-by-Step)

### Step 1: Input Validation

```cpp
if pieces is empty:
    return ""
```

### Step 2: Validate IDs

```cpp
for each id in ids:
    if id < 0 or id >= vocab_size:
        return error
```

### Step 3: Decode Pieces

For each piece:

```cpp
DecodeSentencePiece(piece, id, is_bos_ws):
    if is_control(id):
        return ""  // Invisible

    if is_unknown(id):
        if piece == unk_piece:
            return unk_surface  // Default: "   "
        else:
            return piece  // User-provided unknown text

    // Remove leading ▁ if this is the beginning
    if is_bos_ws and piece starts with ▁:
        piece.remove_prefix(▁)
        if remove_extra_whitespaces:
            # Don't mark as BOS whitespace
            has_bos_ws = false
        else:
            has_bos_ws = true

    # Replace remaining ▁ with space
    piece = piece.replace(▁, " ")

    return (piece, has_bos_ws)
```

### Step 4: Process Byte Sequences

If `byte_fallback` is enabled, consecutive byte pieces are reconstructed:

```cpp
ProcessBytePieces(start, end):
    bytes = []
    for i from start to end:
        byte = PieceToByte(pieces[i])
        bytes.append(byte)

    offset = 0
    while offset < bytes.length:
        consumed = one_utf8_char_length(bytes[offset:])
        if is_valid_utf8(bytes[offset:offset+consumed]):
            # Last byte piece shows the character
            for j in range(consumed):
                if j == consumed - 1:
                    surfaces[start+offset+j] = decode_utf8(bytes[offset:offset+consumed])
                else:
                    surfaces[start+offset+j] = ""
        else:
            # Invalid UTF-8 → replacement character
            surfaces[start+offset] = "�"

        offset += consumed
```

### Step 5: Concatenate Surfaces

```cpp
output = ""
for each piece in pieces:
    output += piece.surface
```

### Step 6: Denormalization (if available)

```cpp
if denormalizer is not null:
    output = denormalizer.Normalize(output)
```

### Step 7: Return Result

```cpp
return output
```

---

## N-Best Encoding

### What is N-Best?

N-Best returns the top N possible tokenizations, sorted by score.

### Algorithm

1. **Build lattice** (same as standard encoding)
2. **Use priority queue** to explore paths
3. **Apply Gumbel noise** (if sampling) for stochastic N-Best

### Key Challenges

- **Memory management**: N-Best can generate many hypotheses
- **Pruning**: Aggressive pruning is needed for efficiency
- **Agenda shrinking**: Reduce hypothesis count periodically

### Implementation Details

The N-Best algorithm uses an "agenda" (priority queue) of hypotheses:

```cpp
struct Hypothesis {
    Node* node;
    Hypothesis* prev;
    float fx;  // Priority (f = g + h)
    float gx;  // Path score so far
}

NBest(nbest_size, sample, theta):
    agenda = priority_queue()
    agenda.push(hypothesis with BOS)

    while len(results) < nbest_size and agenda not empty:
        hyp = agenda.pop()
        if hyp.node == EOS:
            results.append(hyp)
            continue

        for each next_node in begin_nodes[hyp.node.pos]:
            new_score = hyp.gx + theta * next_node.score
            if sampling:
                # Add Gumbel noise for stochastic N-Best
                new_score += -log(-log(uniform(0,1)))

            new_hyp = create hypothesis with next_node
            agenda.push(new_hyp)

        # Prune agenda if too large
        if agenda.size() > MAX_AGENDA_SIZE:
            shrink_agenda(agenda)

    return top nbest_size results
```

---

## Sampling / N-Best Encoding

### Sampling Methods

1. **SampleEncode**: Sample one tokenization according to model probabilities
2. **SampleEncodeAndScore**: Sample multiple tokenizations with inclusion probabilities
3. **NBest with sampling**: Sample from N-best candidates

### Sampling Algorithm

```cpp
Sample(inv_theta):
    alpha = ForwardAlgorithm(inv_theta)  # Forward probabilities
    node = EOS
    results = []

    while node != BOS:
        probs = []
        for each lnode in end_nodes[node.pos]:
            prob = exp(alpha[lnode] + inv_theta * lnode.score - alpha[EOS])
            probs.append(prob)

        # Sample according to probabilities
        node = sample(end_nodes[node.pos], probs)
        results.append(node)

    reverse(results)
    return results
```

### SampleEncodeAndScore

Samples multiple tokenizations and computes inclusion probabilities:

```cpp
SampleEncodeAndScore(inv_theta, num_samples, wor, include_best):
    if wor:  # Without replacement
        # Compute inclusion probabilities
        probs = compute_path_probabilities(inv_theta)
        inclusion_probs = compute_inclusion_probabilities(probs)

        samples = sample_without_replacement(inclusion_probs, num_samples)
        scores = log(inclusion_probs[samples])
    else:  # With replacement
        samples = []
        for i in range(num_samples):
            samples.append(Sample(inv_theta))
        scores = compute_log_probs(samples, inv_theta)

    if include_best:
        # Ensure best path is included
        best = Viterbi()
        if best not in samples:
            samples.insert(best, 0)
            scores.insert(0, 0)  # log(1) = 0

    return (samples, scores)
```

### Inclusion Probability

Probability that a particular tokenization appears in the sample:

```cpp
inclusion_prob[token] = sum(prob[token appears as sample i] for i in 1..num_samples)
```

For sampling without replacement (WOR), this requires careful computation to avoid duplicates.

---

## Important Edge Cases

### 1. Empty Input

- **Encode**: Returns empty result
- **Decode**: Returns empty string

### 2. All Unknown Characters

If all characters are unknown (not in vocabulary):

- Each character becomes a separate UNK token
- Example: "xyz" → ["<unk>", "<unk>", "<unk>"]

### 3. Single Path Segmentation

When there's only one possible tokenization:

- NBest returns only one result
- Sampling returns the same result deterministically
- Regression test for segfault (issue #1198)

### 4. User-Defined Symbol Conflicts

If user-defined symbols overlap:

- Longest match wins (PrefixMatcher behavior)
- User-defined symbols always get bonus score

### 5. Byte Fallback with Invalid UTF-8

If byte fallback produces invalid UTF-8:

- Invalid sequences become U+FFFD (replacement character)
- Each invalid byte produces one replacement character

### 6. Normalization Changes Text Length

If normalization changes the text length (e.g., "㍿" → "株式会社"):

- Alignment mapping (norm_to_orig) tracks this
- Token positions correctly map back to original text

### 7. Mixed Whitespace Markers

If text has both spaces and ▁:

- Spaces are escaped to ▁ during normalization
- During decoding, all ▁ are converted back to spaces

### 8. Consecutive Control Tokens

If multiple control tokens appear:

- All have empty surface
- They don't affect decoded text

### 9. Unused Tokens

If a token is marked as UNUSED:

- It's never added to the lattice
- It's never available for encoding
- Can be used to disable tokens at runtime

### 10. Very Long Text

For very long text (> 4192 bytes by default):

- May be truncated or cause overflow
- Training has `max_sentence_length` limit
- Runtime may have performance issues

---

## Model Protobuf Fields that Matter for Runtime

### ModelProto

- `pieces[]` - Vocabulary (critical)
- `trainer_spec.unk_id` - Unknown token ID
- `trainer_spec.bos_id` - BOS token ID
- `trainer_spec.eos_id` - EOS token ID
- `trainer_spec.pad_id` - PAD token ID
- `trainer_spec.unk_piece` - Unknown token string
- `trainer_spec.bos_piece` - BOS token string
- `trainer_spec.eos_piece` - EOS token string
- `trainer_spec.pad_piece` - PAD token string
- `trainer_spec.unk_surface` - Unknown surface string
- `trainer_spec.byte_fallback` - Enable byte fallback
- `trainer_spec.user_defined_symbols[]` - User-defined symbols
- `trainer_spec.control_symbols[]` - Control symbols
- `normalizer_spec.precompiled_charsmap` - Normalization rules
- `normalizer_spec.add_dummy_prefix` - Add dummy prefix
- `normalizer_spec.remove_extra_whitespaces` - Remove extra whitespace
- `normalizer_spec.escape_whitespaces` - Escape spaces
- `denormalizer_spec.precompiled_charsmap` - Denormalization rules
- `self_test_data.samples[]` - Self-test samples (for validation)

### ModelProto.SentencePiece

- `piece` - Token string
- `score` - Log probability
- `type` - Token type (NORMAL, UNKNOWN, CONTROL, USER_DEFINED, BYTE, UNUSED)

---

## Gotchas that a Rust Reimplementation Must Preserve

### 1. Unicode vs UTF-8 Position Handling

The original lattice uses Unicode positions, but the optimized encoder uses UTF-8 byte positions. Both must produce the same tokenization.

**Gotcha**: Converting between Unicode and UTF-8 positions is tricky, especially with multi-byte characters.

### 2. Whitespace Marker Conventions

- Spaces are escaped to '▁' during normalization
- '▁' at the beginning may be stripped during decoding (dummy prefix)
- Not all '▁' are replaced with ' ' (depends on `add_dummy_prefix`)

**Gotcha**: The logic for stripping the leading '▁' is subtle and depends on multiple flags.

### 3. Unknown Sequence Merging

Consecutive unknown tokens are merged during decoding for better copy/generation.

**Gotcha**: Merging logic must be identical to preserve round-trip accuracy.

### 4. Byte Fallback Surface Mapping

Only the last byte piece in a multi-byte sequence shows the character.

**Gotcha**: This is non-obvious and critical for correct decoding.

### 5. User-Defined Symbol Bonus

User-defined symbols get a large bonus score to ensure selection.

**Gotcha**: The bonus must be large enough to always win, but not so large it causes overflow.

### 6. Viterbi Tie-Breaking

When multiple paths have the same score, the first one found is returned.

**Gotcha**: This must be deterministic for reproducibility.

### 7. Log-Sum-Exp Numerical Stability

Use the log-sum-exp trick to avoid overflow when exponentiating.

**Gotcha**: Direct computation of `log(exp(x) + exp(y))` will overflow.

### 8. Sampling Without Replacement (WOR)

When sampling multiple tokenizations, probabilities must be adjusted to avoid duplicates.

**Gotcha**: The correction factor is `prob[i] / (1 - sum(prev_probs))`.

### 9. Normalization Alignment

The `norm_to_orig` mapping must correctly handle all normalization changes.

**Gotcha**: One-to-many and many-to-one normalizations complicate alignment.

### 10. Trie Search Limits

The `commonPrefixSearch` has a limit on the number of results.

**Gotcha**: If the limit is too low, some matches may be missed.

### 11. Float Comparison

Viterbi uses `>` for score comparison (strictly greater).

**Gotcha**: Use of `>=` would change the tie-breaking behavior.

### 12. Empty Token Handling

Empty tokens (piece == "") are invalid and cause errors.

**Gotcha**: The encoder should reject empty tokens.

### 13. ID Bounds Checking

Piece IDs must be within [0, vocab_size).

**Gotcha**: Out-of-bounds IDs should return errors, not crash.

### 14. Control Token Invisibility

Control tokens have empty surface and don't affect decoded text.

**Gotcha**: This is different from unknown tokens, which have non-empty surface.

### 15. Score Summation Order

Path scores are summed in the order tokens appear.

**Gotcha**: Floating-point addition is not associative; order matters.

### 16. Self-Test Validation

Models may have embedded self-test samples.

**Gotcha**: These should be validated after loading the model.

### 17. Trie Build Order

Trie requires pieces to be sorted before building.

**Gotcha**: Unsorted input will cause build failures.

### 18. Memory Management

The original uses a custom FreeList for efficient node allocation.

**Gotcha**: In Rust, you'll need a different allocation strategy.

### 19. Random Number Generation

Sampling uses a random number generator.

**Gotcha**: For reproducibility, the seed must be settable.

### 20. Endianness Handling

The precompiled charsmap may be stored in big-endian format.

**Gotcha**: Must convert to native endianness on load.

---

## Golden Test Strategy for Comparing Against Official SentencePiece

### Test Categories

1. **Basic Encoding/Decoding**
   - Simple English sentences
   - Mixed-case text
   - Punctuation
   - Numbers

2. **Unicode Handling**
   - Multi-byte characters (Chinese, Japanese, Arabic, etc.)
   - Emoji
   - Combining characters
   - Invalid UTF-8

3. **Edge Cases**
   - Empty input
   - All unknown characters
   - All known characters
   - Single character
   - Very long text

4. **Whitespace Handling**
   - Leading whitespace
   - Trailing whitespace
   - Multiple spaces
   - Space-only text
   - Mixed whitespace (tabs, newlines)

5. **Special Tokens**
   - BOS/EOS
   - Unknown tokens
   - Control tokens
   - User-defined symbols
   - Byte fallback tokens

6. **Normalization**
   - Full-width to half-width
   - One-to-many mappings (㍿ → 株式会社)
   - Case folding (if applicable)
   - NFKC normalization

7. **N-Best and Sampling**
   - NBest encoding (n=1, n=5, n=10)
   - Sample encoding (various inv_theta values)
   - SampleEncodeAndScore (with and without replacement)

8. **Round-Trip Tests**
   - Encode → Decode → Compare with original
   - Decode → Encode → Compare with original tokens
   - Test with and without special tokens

### Test Data Format

```rust
struct TestCase {
    name: String,
    input: String,
    expected_pieces: Option<Vec<String>>,  // For encoding tests
    expected_ids: Option<Vec<i32>>,       // For encoding tests
    expected_output: Option<String>,      // For decoding tests
    options: TestOptions,
}

struct TestOptions {
    add_bos: bool,
    add_eos: bool,
    sample: bool,
    nbest: usize,
    inv_theta: f32,
}
```

### Comparison Strategy

1. **Exact String Comparison**
   - For token strings and decoded text
   - Must match exactly

2. **Score Comparison**
   - For NBest and sampling
   - Use approximate equality (epsilon = 1e-6)

3. **Equivalence Testing**
   - Use `VerifyOutputsEquivalent` when available
   - Compares scores, not just strings

4. **Behavioral Testing**
   - Test error conditions
   - Test edge cases
   - Test resource limits

### Automated Test Generation

1. **Extract Test Cases from Official Tests**
   - Parse `unigram_model_test.cc`
   - Parse `sentencepiece_processor_test.cc`
   - Convert to Rust test format

2. **Generate Random Tests**
   - Random strings from vocabulary
   - Random combinations of special tokens
   - Random lengths and patterns

3. **Property-Based Testing**
   - Round-trip property: decode(encode(text)) ≈ text
   - Consistency property: encode(text) should be consistent across runs
   - Monotonicity property: higher scores should be preferred

### Test Execution Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use sentencepiece_sys as sp;  // FFI bindings to official C++ library

    fn test_encode_official_vs_rust(input: &str) {
        // Get official results
        let official_pieces = sp::encode_as_pieces(input);
        let official_ids = sp::encode_as_ids(input);

        // Get Rust results
        let rust_pieces = rust_encode_as_pieces(input);
        let rust_ids = rust_encode_as_ids(input);

        // Compare
        assert_eq!(official_pieces, rust_pieces, "Pieces don't match for: {}", input);
        assert_eq!(official_ids, rust_ids, "IDs don't match for: {}", input);
    }

    fn test_decode_official_vs_rust(pieces: &[&str]) {
        // Get official results
        let official_text = sp::decode_pieces(pieces);

        // Get Rust results
        let rust_text = rust_decode_pieces(pieces);

        // Compare
        assert_eq!(official_text, rust_text, "Decoded text doesn't match");
    }
}
```

### Continuous Testing

1. **Run tests on every model file**
   - Test with various official `.model` files
   - Test with models trained with different settings

2. **Regression Testing**
   - Run after every change
   - Flag any behavior changes

3. **Performance Testing**
   - Benchmark against official implementation
   - Target similar or better performance

---

## Notes for Future WASM/Browser Compatibility

### Current Dependencies

The C++ implementation has dependencies that may not work in WASM:

1. **Filesystem Access**
   - `#include <filesystem>`
   - Uses `std::fstream` for file I/O
   - **WASM Solution**: Use in-memory model loading only

2. **Threading**
   - Uses `std::thread` for parallel processing
   - **WASM Solution**: Single-threaded execution only

3. **Random Number Generation**
   - Uses `std::mt19937` and `std::random_device`
   - **WASM Solution**: Use WASM-compatible RNG (e.g., `wasm-bindgen`)

4. **Dynamic Memory Allocation**
   - Uses custom allocators (FreeList)
   - **WASM Solution**: Use Rust's standard allocator

### WASM-Specific Considerations

1. **Memory Limits**
   - WASM has limited memory
   - **Solution**: Use streaming decoding, limit model size

2. **Performance**
   - WASM may be slower than native
   - **Solution**: Optimize hot paths, use SIMD if available

3. **JavaScript Interop**
   - Need to convert between JS strings and Rust strings
   - **Solution**: Use `wasm-bindgen` with `JsString`

4. **Error Handling**
   - C++ uses exceptions (in some places)
   - **Solution**: Use `Result<T, E>` in Rust

### Recommended WASM Architecture

```rust
#[wasm_bindgen]
pub struct SentencePieceProcessor {
    model: UnigramModel,
}

#[wasm_bindgen]
impl SentencePieceProcessor {
    #[wasm_bindgen(constructor)]
    pub fn new(model_data: &[u8]) -> Result<SentencePieceProcessor, JsError> {
        // Load model from byte array (no filesystem needed)
        let model = UnigramModel::load(model_data)?;
        Ok(SentencePieceProcessor { model })
    }

    #[wasm_bindgen]
    pub fn encode(&self, text: &str) -> Result<JsValue, JsError> {
        let pieces = self.model.encode(text)?;
        // Convert to JavaScript array
        Ok(serde_wasm_bindgen::to_value(&pieces)?)
    }

    #[wasm_bindgen]
    pub fn decode(&self, pieces: &[String]) -> Result<String, JsError> {
        let text = self.model.decode(pieces)?;
        Ok(text)
    }
}
```

### WASM-Specific Optimizations

1. **Avoid heap allocations in hot paths**
2. **Use `#[inline]` for small functions**
3. **Pre-allocate buffers where possible**
4. **Use `#[no_mangle]` for critical functions**

### Testing in WASM

```rust
#[wasm_bindgen_test]
fn test_encode_basic() {
    let model_data = include_bytes!("test.model");
    let processor = SentencePieceProcessor::new(model_data).unwrap();

    let result = processor.encode("Hello world").unwrap();
    assert_eq!(result, vec!["▁Hello", "▁world"]);
}
```

---

## Checklist for Future Rust Implementation

### Phase 1: Core Data Structures

- [ ] Implement ModelProto parsing (protobuf deserialization)
- [ ] Implement vocabulary storage (pieces array)
- [ ] Implement prefix trie (or use a crate like `cedar` or `daachorse`)
- [ ] Implement Normalizer with trie-based longest match
- [ ] Implement PrefixMatcher for user-defined symbols
- [ ] Implement Unicode/UTF-8 utilities (length detection, validation)

### Phase 2: Normalization

- [ ] Implement `Normalize()` function
- [ ] Implement `NormalizePrefix()` function
- [ ] Implement alignment mapping (norm_to_orig)
- [ ] Handle add_dummy_prefix
- [ ] Handle remove_extra_whitespaces
- [ ] Handle escape_whitespaces
- [ ] Support precompiled charsmap loading

### Phase 3: Lattice and Viterbi

- [ ] Implement Lattice data structure
- [ ] Implement Node structure
- [ ] Implement Lattice::SetSentence()
- [ ] Implement Lattice::Insert()
- [ ] Implement PopulateNodes()
- [ ] Implement Lattice::Viterbi()
- [ ] Implement ForwardAlgorithm() (for training/sampling)

### Phase 4: Model Implementation

- [ ] Implement UnigramModel::Encode()
- [ ] Implement UnigramModel::EncodeOptimized()
- [ ] Implement UnigramModel::NBestEncode()
- [ ] Implement UnigramModel::SampleEncode()
- [ ] Implement UnigramModel::PieceToId()
- [ ] Implement UnigramModel::IdToPiece()
- [ ] Implement type checking methods (IsUnknown, IsControl, etc.)

### Phase 5: Processor

- [ ] Implement SentencePieceProcessor
- [ ] Implement SentencePieceProcessor::Load()
- [ ] Implement SentencePieceProcessor::Encode()
- [ ] Implement SentencePieceProcessor::Decode()
- [ ] Implement extra options (bos, eos, reverse)
- [ ] Implement error handling

### Phase 6: Decoding

- [ ] Implement DecodeSentencePiece()
- [ ] Implement byte fallback decoding
- [ ] Implement unknown sequence merging
- [ ] Implement whitespace marker handling
- [ ] Implement denormalization
- [ ] Handle control tokens

### Phase 7: Advanced Features

- [ ] Implement NBest with sampling
- [ ] Implement SampleEncodeAndScore
- [ ] Implement CalculateEntropy
- [ ] Implement vocabulary constraints (SetVocabulary)
- [ ] Implement self-test validation

### Phase 8: Testing

- [ ] Port tests from `unigram_model_test.cc`
- [ ] Port tests from `sentencepiece_processor_test.cc`
- [ ] Port tests from `normalizer_test.cc`
- [ ] Add round-trip tests
- [ ] Add property-based tests
- [ ] Add WASM-specific tests

### Phase 9: Optimization

- [ ] Profile and optimize hot paths
- [ ] Consider SIMD for UTF-8 processing
- [ ] Optimize memory allocations
- [ ] Implement caching where beneficial
- [ ] Benchmark against C++ implementation

### Phase 10: Documentation and Examples

- [ ] Write API documentation
- [ ] Write usage examples
- [ ] Write migration guide from C++ to Rust
- [ ] Document performance characteristics
- [ ] Document differences from C++ implementation

### Phase 11: WASM Support

- [ ] Add `wasm-bindgen` bindings
- [ ] Implement in-memory model loading
- [ ] Optimize for WASM
- [ ] Write WASM-specific tests
- [ ] Create npm package

### Phase 12: Integration

- [ ] Create Python bindings (optional)
- [ ] Create Node.js bindings (optional)
- [ ] Add to crates.io
- [ ] Write CI/CD pipeline
- [ ] Write migration guide for users

---

## References and Further Reading

### Official Documentation

- [SentencePiece GitHub](https://github.com/google/sentencepiece)
- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)
- [Unigram Language Model Paper](https://arxiv.org/abs/1804.10959)

### Key Algorithms

- [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
- [Forward-Backward Algorithm](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm)
- [Double-Array Trie](https://linux.thai.net/~thep/datrie/datrie.html)
- [Log-Sum-Exp Trick](https://en.wikipedia.org/wiki/LogSumExp)

### Sampling Methods

- [Gumbel-Max Trick](https://arxiv.org/abs/1611.01144)
- [Nucleus Sampling](https://arxiv.org/abs/1904.09751)
- [Top-k Sampling](https://arxiv.org/abs/1904.09751)

### UTF-8 and Unicode

- [UTF-8 Wikipedia](https://en.wikipedia.org/wiki/UTF-8)
- [Unicode Standard](https://unicode.org/standard/standard.html)
- [Unicode Normalization](https://unicode.org/reports/tr15/)

### Rust Crates That May Help

- `prost` - Protocol Buffers
- `cedar` or `daachorse` - Double-array tries
- `unicode-normalization` - Unicode normalization
- `rand` - Random number generation
- `wasm-bindgen` - WASM bindings
- `serde` - Serialization (for testing)

---

## Conclusion

This guide captures the essential implementation details needed to create a clean-room Rust reimplementation of the SentencePiece Unigram tokenizer runtime. The key challenges are:

1. **Correctness**: Must produce identical results to the C++ implementation
2. **Performance**: Should be competitive with the optimized C++ encoder
3. **Compatibility**: Should work with existing `.model` files
4. **Portability**: Should work in WASM and other environments

The implementation should be tested extensively against the official C++ library to ensure correctness. The optimization techniques (especially EncodeOptimized) are critical for performance.

Good luck with the implementation!
