#ifndef TYPES_H
#define TYPES_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long int a64;

// This code is initially optimized for nVidia RTX 4090
const u32 BLOCK_LOG2    = 8;                         // log2 of block size
const u32 BLOCK_SIZE    = 1 << BLOCK_LOG2;           // threads per block
const u32 TOTAL_BLOCKS  = BLOCK_SIZE;                // must be = BLOCK_SIZE
const u32 TOTAL_THREADS = TOTAL_BLOCKS * BLOCK_SIZE; // total threads
const u32 UNIT_SIZE     = 4;                         // threads per rewrite unit
const u32 TOTAL_UNITS   = TOTAL_THREADS / UNIT_SIZE; // total rewrite units
const u32 NODE_SIZE     = 1 << 28;                   // total nodes (2GB addressable)
const u32 BAGS_SIZE     = TOTAL_BLOCKS * BLOCK_SIZE; // total parallel redexes
const u32 GTMP_SIZE     = BLOCK_SIZE;                // TODO: rename
const u32 GIDX_SIZE     = BAGS_SIZE;                 // TODO: rename
const u32 ALLOC_PAD     = NODE_SIZE / TOTAL_UNITS;   // space between unit alloc areas

// Types
// -----

// Pointer value (28-bit)
typedef u32 Val;

// Pointer tags (4-bit)
typedef u8 Tag;
const Tag NIL = 0x0; // empty node
const Tag REF = 0x1; // reference to a definition (closed net)
const Tag NUM = 0x2; // unboxed number
const Tag ERA = 0x3; // unboxed eraser
const Tag VRR = 0x4; // variable pointing to root
const Tag VR1 = 0x5; // variable pointing to aux1 port of node
const Tag VR2 = 0x6; // variable pointing to aux2 port of node
const Tag RDR = 0x7; // redirection to root
const Tag RD1 = 0x8; // redirection to aux1 port of node
const Tag RD2 = 0x9; // redirection to aux2 port of node
const Tag CON = 0xA; // points to main port of con node
const Tag DUP = 0xB; // points to main port of dup node
const Tag TRI = 0xC; // points to main port of tri node
const Tag QUA = 0xD; // points to main port of qua node
const Tag QUI = 0xE; // points to main port of qui node
const Tag SEX = 0xF; // points to main port of sex node
const u32 NEO = 0xFFFFFFFD; // recently allocated value
const u32 GON = 0xFFFFFFFE; // node has been moved to redex bag
const u32 BSY = 0xFFFFFFFF; // value taken by another thread, will be replaced soon

// Rewrite fractions
const u32 A1 = 0;
const u32 A2 = 1;
const u32 B1 = 2;
const u32 B2 = 3;

// Ports (P1 or P2)
typedef u8 Port;
const u32 P1 = 0;
const u32 P2 = 1;

// Pointers = 4-bit tag + 28-bit val
typedef u32 Ptr;

// Nodes are pairs of pointers
typedef struct alignas(8) {
  Ptr ports[2];
} Node;

// Wires are pairs of pointers
typedef struct alignas(8) {
  Ptr lft;
  Ptr rgt;
} Wire;

// An interaction net 
typedef struct {
  Ptr   root; // root wire
  u32   blen; // total bag length (redex count)
  Wire* bags; // redex bags (active pairs)
  Node* node; // memory buffer with all nodes
  u32*  gidx; // ............
  u32*  gtmp; // aux obj for communication
  u32   done; // number of completed threads
  u32   rwts; // number of rewrites performed
} Net;

// A worker local data
typedef struct {
  u32   tid;   // thread id
  u32   bid;   // block id 
  u32   gid;   // global id
  u32   unit;  // unit id (index on redex array)
  u32   frac;  // worker frac (A1|A2|B1|B2)
  u32   port;  // worker port (P1|P2)
  Ptr   a_ptr; // left pointer of active wire
  Ptr   b_ptr; // right pointer of active wire
  u32   aloc;  // where to alloc next node
  u32   rwts;  // total rewrites this performed
  Wire* bag;   // local redex bag
} Worker;

typedef struct {
  u32 keys[65536];
  u32 vals[65536];
  u32 size;
} Map;

#endif
