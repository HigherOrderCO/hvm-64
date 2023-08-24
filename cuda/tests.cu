#include "tests.cuh"

// term_a = (λx(x) λy(y))
__host__ void inject_term_a(Net* net) {
  net->root     = 0x60000001;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000001};
  net->node[ 0] = (Node) {0x60000000,0x50000000};
  net->node[ 1] = (Node) {0xa0000002,0x40000000};
  net->node[ 2] = (Node) {0x60000002,0x50000002};
}

// term_b = (λfλx(f x) λy(y))
__host__ void inject_term_b(Net* net) {
  net->root     = 0x60000003;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000003};
  net->node[ 0] = (Node) {0xa0000001,0xa0000002};
  net->node[ 1] = (Node) {0x50000002,0x60000002};
  net->node[ 2] = (Node) {0x50000001,0x60000001};
  net->node[ 3] = (Node) {0xa0000004,0x40000000};
  net->node[ 4] = (Node) {0x60000004,0x50000004};
}

// term_c = (λfλx(f (f x)) λx(x))
__host__ void inject_term_c(Net* net) {
  net->root     = 0x60000005;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node[ 2] = (Node) {0x50000004,0x50000003};
  net->node[ 3] = (Node) {0x60000002,0x60000004};
  net->node[ 4] = (Node) {0x50000002,0x60000003};
  net->node[ 5] = (Node) {0xa0000006,0x40000000};
  net->node[ 6] = (Node) {0x60000006,0x50000006};
}

// term_d = (λfλx(f (f x)) λgλy(g (g y)))
__host__ void inject_term_d(Net* net) {
  net->root     = 0x60000005;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node[ 2] = (Node) {0x50000004,0x50000003};
  net->node[ 3] = (Node) {0x60000002,0x60000004};
  net->node[ 4] = (Node) {0x50000002,0x60000003};
  net->node[ 5] = (Node) {0xa0000006,0x40000000};
  net->node[ 6] = (Node) {0xc0000007,0xa000000a};
  net->node[ 7] = (Node) {0xa0000008,0xa0000009};
  net->node[ 8] = (Node) {0x5000000a,0x50000009};
  net->node[ 9] = (Node) {0x60000008,0x6000000a};
  net->node[10] = (Node) {0x50000008,0x60000009};
}

// term_e = (c2 g_s g_z)
__host__ void inject_term_e(Net* net) {
  net->root     = 0x6000000b;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node[ 2] = (Node) {0x50000004,0x50000003};
  net->node[ 3] = (Node) {0x60000002,0x60000004};
  net->node[ 4] = (Node) {0x50000002,0x60000003};
  net->node[ 5] = (Node) {0xa0000006,0xa000000b};
  net->node[ 6] = (Node) {0xc0000007,0xa0000008};
  net->node[ 7] = (Node) {0x50000009,0x5000000a};
  net->node[ 8] = (Node) {0xa0000009,0x6000000a};
  net->node[ 9] = (Node) {0x50000007,0xa000000a};
  net->node[10] = (Node) {0x60000007,0x60000008};
  net->node[11] = (Node) {0xa000000c,0x40000000};
  net->node[12] = (Node) {0x6000000c,0x5000000c};
}

// term_f = (c3 g_s g_z)
__host__ void inject_term_f(Net* net) {
  net->root     = 0x6000000d;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000007};
  net->node[ 0] = (Node) {0xb0000001,0xa0000006};
  net->node[ 1] = (Node) {0xb0000002,0xa0000005};
  net->node[ 2] = (Node) {0xa0000003,0xa0000004};
  net->node[ 3] = (Node) {0x50000006,0x50000004};
  net->node[ 4] = (Node) {0x60000003,0x50000005};
  net->node[ 5] = (Node) {0x60000004,0x60000006};
  net->node[ 6] = (Node) {0x50000003,0x60000005};
  net->node[ 7] = (Node) {0xa0000008,0xa000000d};
  net->node[ 8] = (Node) {0xc0000009,0xa000000a};
  net->node[ 9] = (Node) {0x5000000b,0x5000000c};
  net->node[10] = (Node) {0xa000000b,0x6000000c};
  net->node[11] = (Node) {0x50000009,0xa000000c};
  net->node[12] = (Node) {0x60000009,0x6000000a};
  net->node[13] = (Node) {0xa000000e,0x40000000};
  net->node[14] = (Node) {0x6000000e,0x5000000e};
}

// term_g = (c8 g_s g_z)
__host__ void inject_term_g(Net* net) {
  net->root     = 0x60000017;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000011};
  net->node[ 0] = (Node) {0xb0000001,0xa0000010};
  net->node[ 1] = (Node) {0xb0000002,0xa000000f};
  net->node[ 2] = (Node) {0xb0000003,0xa000000e};
  net->node[ 3] = (Node) {0xb0000004,0xa000000d};
  net->node[ 4] = (Node) {0xb0000005,0xa000000c};
  net->node[ 5] = (Node) {0xb0000006,0xa000000b};
  net->node[ 6] = (Node) {0xb0000007,0xa000000a};
  net->node[ 7] = (Node) {0xa0000008,0xa0000009};
  net->node[ 8] = (Node) {0x50000010,0x50000009};
  net->node[ 9] = (Node) {0x60000008,0x5000000a};
  net->node[10] = (Node) {0x60000009,0x5000000b};
  net->node[11] = (Node) {0x6000000a,0x5000000c};
  net->node[12] = (Node) {0x6000000b,0x5000000d};
  net->node[13] = (Node) {0x6000000c,0x5000000e};
  net->node[14] = (Node) {0x6000000d,0x5000000f};
  net->node[15] = (Node) {0x6000000e,0x60000010};
  net->node[16] = (Node) {0x50000008,0x6000000f};
  net->node[17] = (Node) {0xa0000012,0xa0000017};
  net->node[18] = (Node) {0xc0000013,0xa0000014};
  net->node[19] = (Node) {0x50000015,0x50000016};
  net->node[20] = (Node) {0xa0000015,0x60000016};
  net->node[21] = (Node) {0x50000013,0xa0000016};
  net->node[22] = (Node) {0x60000013,0x60000014};
  net->node[23] = (Node) {0xa0000018,0x40000000};
  net->node[24] = (Node) {0x60000018,0x50000018};
}

// term_h = (c12 g_s g_z)
__host__ void inject_term_h(Net* net) {
  net->root     = 0x6000001f;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000019};
  net->node[ 0] = (Node) {0xb0000001,0xa0000018};
  net->node[ 1] = (Node) {0xb0000002,0xa0000017};
  net->node[ 2] = (Node) {0xb0000003,0xa0000016};
  net->node[ 3] = (Node) {0xb0000004,0xa0000015};
  net->node[ 4] = (Node) {0xb0000005,0xa0000014};
  net->node[ 5] = (Node) {0xb0000006,0xa0000013};
  net->node[ 6] = (Node) {0xb0000007,0xa0000012};
  net->node[ 7] = (Node) {0xb0000008,0xa0000011};
  net->node[ 8] = (Node) {0xb0000009,0xa0000010};
  net->node[ 9] = (Node) {0xb000000a,0xa000000f};
  net->node[10] = (Node) {0xb000000b,0xa000000e};
  net->node[11] = (Node) {0xa000000c,0xa000000d};
  net->node[12] = (Node) {0x50000018,0x5000000d};
  net->node[13] = (Node) {0x6000000c,0x5000000e};
  net->node[14] = (Node) {0x6000000d,0x5000000f};
  net->node[15] = (Node) {0x6000000e,0x50000010};
  net->node[16] = (Node) {0x6000000f,0x50000011};
  net->node[17] = (Node) {0x60000010,0x50000012};
  net->node[18] = (Node) {0x60000011,0x50000013};
  net->node[19] = (Node) {0x60000012,0x50000014};
  net->node[20] = (Node) {0x60000013,0x50000015};
  net->node[21] = (Node) {0x60000014,0x50000016};
  net->node[22] = (Node) {0x60000015,0x50000017};
  net->node[23] = (Node) {0x60000016,0x60000018};
  net->node[24] = (Node) {0x5000000c,0x60000017};
  net->node[25] = (Node) {0xa000001a,0xa000001f};
  net->node[26] = (Node) {0xc000001b,0xa000001c};
  net->node[27] = (Node) {0x5000001d,0x5000001e};
  net->node[28] = (Node) {0xa000001d,0x6000001e};
  net->node[29] = (Node) {0x5000001b,0xa000001e};
  net->node[30] = (Node) {0x6000001b,0x6000001c};
  net->node[31] = (Node) {0xa0000020,0x40000000};
  net->node[32] = (Node) {0x60000020,0x50000020};
}

// term_i = (c14 g_s g_z)
__host__ void inject_term_i(Net* net) {
  net->root     = 0x60000023;
  net->bags[ 0] = (Wire) {0xa0000000,0xa000001d};
  net->node[ 0] = (Node) {0xb0000001,0xa000001c};
  net->node[ 1] = (Node) {0xb0000002,0xa000001b};
  net->node[ 2] = (Node) {0xb0000003,0xa000001a};
  net->node[ 3] = (Node) {0xb0000004,0xa0000019};
  net->node[ 4] = (Node) {0xb0000005,0xa0000018};
  net->node[ 5] = (Node) {0xb0000006,0xa0000017};
  net->node[ 6] = (Node) {0xb0000007,0xa0000016};
  net->node[ 7] = (Node) {0xb0000008,0xa0000015};
  net->node[ 8] = (Node) {0xb0000009,0xa0000014};
  net->node[ 9] = (Node) {0xb000000a,0xa0000013};
  net->node[10] = (Node) {0xb000000b,0xa0000012};
  net->node[11] = (Node) {0xb000000c,0xa0000011};
  net->node[12] = (Node) {0xb000000d,0xa0000010};
  net->node[13] = (Node) {0xa000000e,0xa000000f};
  net->node[14] = (Node) {0x5000001c,0x5000000f};
  net->node[15] = (Node) {0x6000000e,0x50000010};
  net->node[16] = (Node) {0x6000000f,0x50000011};
  net->node[17] = (Node) {0x60000010,0x50000012};
  net->node[18] = (Node) {0x60000011,0x50000013};
  net->node[19] = (Node) {0x60000012,0x50000014};
  net->node[20] = (Node) {0x60000013,0x50000015};
  net->node[21] = (Node) {0x60000014,0x50000016};
  net->node[22] = (Node) {0x60000015,0x50000017};
  net->node[23] = (Node) {0x60000016,0x50000018};
  net->node[24] = (Node) {0x60000017,0x50000019};
  net->node[25] = (Node) {0x60000018,0x5000001a};
  net->node[26] = (Node) {0x60000019,0x5000001b};
  net->node[27] = (Node) {0x6000001a,0x6000001c};
  net->node[28] = (Node) {0x5000000e,0x6000001b};
  net->node[29] = (Node) {0xa000001e,0xa0000023};
  net->node[30] = (Node) {0xc000001f,0xa0000020};
  net->node[31] = (Node) {0x50000021,0x50000022};
  net->node[32] = (Node) {0xa0000021,0x60000022};
  net->node[33] = (Node) {0x5000001f,0xa0000022};
  net->node[34] = (Node) {0x6000001f,0x60000020};
  net->node[35] = (Node) {0xa0000024,0x40000000};
  net->node[36] = (Node) {0x60000024,0x50000024};
}

// term_j = (c18 g_s g_z)
__host__ void inject_term_j(Net* net) {
  net->root    = 0x6000002b;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000025};
  net->node[ 0] = (Node) {0xb0000001,0xa0000024};
  net->node[ 1] = (Node) {0xb0000002,0xa0000023};
  net->node[ 2] = (Node) {0xb0000003,0xa0000022};
  net->node[ 3] = (Node) {0xb0000004,0xa0000021};
  net->node[ 4] = (Node) {0xb0000005,0xa0000020};
  net->node[ 5] = (Node) {0xb0000006,0xa000001f};
  net->node[ 6] = (Node) {0xb0000007,0xa000001e};
  net->node[ 7] = (Node) {0xb0000008,0xa000001d};
  net->node[ 8] = (Node) {0xb0000009,0xa000001c};
  net->node[ 9] = (Node) {0xb000000a,0xa000001b};
  net->node[10] = (Node) {0xb000000b,0xa000001a};
  net->node[11] = (Node) {0xb000000c,0xa0000019};
  net->node[12] = (Node) {0xb000000d,0xa0000018};
  net->node[13] = (Node) {0xb000000e,0xa0000017};
  net->node[14] = (Node) {0xb000000f,0xa0000016};
  net->node[15] = (Node) {0xb0000010,0xa0000015};
  net->node[16] = (Node) {0xb0000011,0xa0000014};
  net->node[17] = (Node) {0xa0000012,0xa0000013};
  net->node[18] = (Node) {0x50000024,0x50000013};
  net->node[19] = (Node) {0x60000012,0x50000014};
  net->node[20] = (Node) {0x60000013,0x50000015};
  net->node[21] = (Node) {0x60000014,0x50000016};
  net->node[22] = (Node) {0x60000015,0x50000017};
  net->node[23] = (Node) {0x60000016,0x50000018};
  net->node[24] = (Node) {0x60000017,0x50000019};
  net->node[25] = (Node) {0x60000018,0x5000001a};
  net->node[26] = (Node) {0x60000019,0x5000001b};
  net->node[27] = (Node) {0x6000001a,0x5000001c};
  net->node[28] = (Node) {0x6000001b,0x5000001d};
  net->node[29] = (Node) {0x6000001c,0x5000001e};
  net->node[30] = (Node) {0x6000001d,0x5000001f};
  net->node[31] = (Node) {0x6000001e,0x50000020};
  net->node[32] = (Node) {0x6000001f,0x50000021};
  net->node[33] = (Node) {0x60000020,0x50000022};
  net->node[34] = (Node) {0x60000021,0x50000023};
  net->node[35] = (Node) {0x60000022,0x60000024};
  net->node[36] = (Node) {0x50000012,0x60000023};
  net->node[37] = (Node) {0xa0000026,0xa000002b};
  net->node[38] = (Node) {0xc0000027,0xa0000028};
  net->node[39] = (Node) {0x50000029,0x5000002a};
  net->node[40] = (Node) {0xa0000029,0x6000002a};
  net->node[41] = (Node) {0x50000027,0xa000002a};
  net->node[42] = (Node) {0x60000027,0x60000028};
  net->node[43] = (Node) {0xa000002c,0x40000000};
  net->node[44] = (Node) {0x6000002c,0x5000002c};
}
