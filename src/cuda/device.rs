use cudarc::driver::DeviceRepr;

//// Constants
// NOTE: Keep in sync with CUDA code

// Bags dimensions (128x128 redex bags)
pub const BAGS_WIDTH_L2: u32 = 7;
pub const BAGS_WIDTH: u32    = 1 << BAGS_WIDTH_L2;
pub const BAGS_HEIGHT_L2: u32 = 7;
pub const BAGS_HEIGHT: u32    = 1 << BAGS_HEIGHT_L2;
pub const BAGS_TOTAL_L2: u32  = BAGS_WIDTH_L2 + BAGS_HEIGHT_L2;
pub const BAGS_TOTAL: u32     = 1 << BAGS_TOTAL_L2;

// Threads per Squad (4)
pub const SQUAD_SIZE_L2: u32 = 2;
pub const SQUAD_SIZE: u32    = 1 << SQUAD_SIZE_L2;

// Squads per Block (128)
pub const GROUP_SIZE_L2: u32 = BAGS_WIDTH_L2;
pub const GROUP_SIZE: u32    = 1 << GROUP_SIZE_L2;

// Threads per Block (512)
pub const BLOCK_SIZE_L2: u32 = GROUP_SIZE_L2 + SQUAD_SIZE_L2;
pub const BLOCK_SIZE: u32    = 1 << BLOCK_SIZE_L2;

// Heap Size (max total nodes = 256m = 2GB)
pub const HEAP_SIZE_L2: u32 = 28;
pub const HEAP_SIZE: u32    = 1 << HEAP_SIZE_L2;

// Jump Table (max book entries = 16m definitions)
pub const JUMP_SIZE_L2: u32 = 24;
pub const JUMP_SIZE: u32    = 1 << JUMP_SIZE_L2;

// Max Redexes per Interaction
pub const MAX_NEW_REDEX: u32 = 16; // FIXME: use to check full rbags

// Local Attributes per Squad
pub const SMEM_SIZE: u32 = 4; // local attributes

// Total Number of Squads
pub const SQUAD_TOTAL_L2: u32 = BAGS_TOTAL_L2;
pub const SQUAD_TOTAL: u32    = 1 << SQUAD_TOTAL_L2;

// Total Allocation Nodes per Squad
pub const AREA_SIZE: u32 = HEAP_SIZE / SQUAD_TOTAL;

// Redexes per Redex Bag
pub const RBAG_SIZE: u32 = 256;

// Total Redexes on All Bags
pub const BAGS_SIZE: u32 = BAGS_TOTAL * RBAG_SIZE;

// Max Global Expansion Ptrs (1 per squad)
pub const HEAD_SIZE_L2: u32 = SQUAD_TOTAL_L2;
pub const HEAD_SIZE: u32    = 1 << HEAD_SIZE_L2;

// Max Local Expansion Ptrs per Squad
pub const EXPANSIONS_PER_SQUAD: u32 = 16;


// Core terms
pub const VR1: Tag = 0x0; // variable to aux port 1
pub const VR2: Tag = 0x1; // variable to aux port 2
// [...]
pub const REF: Tag = 0x4; // lazy closed net
// [...]

// Special values
pub const ROOT: u32 = 0x0 | VR2 as u32;  // pointer to root port


//// Types

// Pointers = 4-bit tag + 28-bit val
pub type Ptr = u32;

// Nodes are pairs of pointers
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Node {
  pub ports: [Ptr; 2],
}

// Wires are pairs of pointers
pub type Wire = u64;

#[repr(C)]
pub struct CudaNet {
  pub bags: *mut Wire, // redex bags (active pairs)
  pub heap: *mut Node, // memory buffer with all nodes
  pub head: *mut Wire, // head expansion buffer
  pub jump: *mut u32 , // book jump table
  pub rwts: u64  , // number of rewrites performed
}

unsafe impl DeviceRepr for Node {}
unsafe impl DeviceRepr for CudaNet {}

pub type Tag = u8 ; // pointer tag: 4-bit
pub type Val = u32; // pointer val: 28-bit


//// Functions

// Creates a new pointer
pub fn mkptr(tag: Tag, val: Val) -> Ptr {
  (val << 4) | (tag as Val)
}

// Gets the value of a pointer
pub fn val(ptr: Ptr) -> Val {
  (ptr >> 4) as Val
}
