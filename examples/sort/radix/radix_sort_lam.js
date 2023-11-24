
//// data Arr = Empty | (Single x) | (Concat x0 x1)
//Empty  =         λempty λsingle λconcat empty
//Single = λx      λempty λsingle λconcat (single x)
//Concat = λx0 λx1 λempty λsingle λconcat (concat x0 x1)

//// data Map = Free | Used | (Node x0 x1)
//Free =         λfree λused λnode free
//Used =         λfree λused λnode used
//Node = λx0 λx1 λfree λused λnode (node x0 x1)

//// gen : u32 -> Arr
//gen = λn match n {
  //0: λx (Single x)
  //1+p: λx
    //let x0 = (<< x 1)
    //let x1 = (| x0 1)
    //(Concat (gen p x0) (gen p x1))
//}

//// rev : Arr -> Arr
//rev = λa
  //let a_empty  = Empty
  //let a_single = λx (Single x)
  //let a_concat = λx0 λx1 (Concat (rev x1) (rev x0))
  //(a a_empty a_single a_concat)

//// sum : Arr -> u32
//sum = λa
  //let a_empty  = 0
  //let a_single = λx x
  //let a_concat = λx0 λx1 (+ (sum x0) (sum x1))
  //(a a_empty a_single a_concat)

//// sort : Arr -> Arr
//sort = λt (to_arr (to_map t) 0)

//// to_arr : Map -> u32 -> Arr
//to_arr = λa
  //let a_free = λk Empty
  //let a_used = λk (Single k)
  //let a_node = λx0 λx1 λk
    //let x0 = (to_arr x0 (+ (* k 2) 0))
    //let x1 = (to_arr x1 (+ (* k 2) 1))
    //(Concat x0 x1)
  //(a a_free a_used a_node)

//// to_map : Arr -> Map
//to_map = λa
  //let a_empty  = Free
  //let a_single = λx (radix x)
  //let a_concat = λx0 λx1 (merge (to_map x0) (to_map x1))
  //(a a_empty a_single a_concat)

//// merge : Map -> Map -> Map
//merge = λa
  //let a_free = λb
    //let b_free = Free
    //let b_used = Used
    //let b_node = λb0 λb1 (Node b0 b1)
    //(b b_free b_used b_node)
  //let a_used = λb
    //let b_free = Used
    //let b_used = Used
    //let b_node = λb0 λb1 0
    //(b b_free b_used b_node)
  //let a_node = λa0 λa1 λb
    //let b_free = λa0 λa1 (Node a0 a1)
    //let b_used = λa0 λa1 0
    //let b_node = λb0 λb1 λa0 λa1 (Node (merge a0 b0) (merge a1 b1))
    //(b b_free b_used b_node a0 a1)
  //(a a_free a_used a_node)

//// radix : u32 -> Map
//radix = λn
  //let r = Used
  //let r = (swap (& n       1) r Free)
  //let r = (swap (& n       2) r Free)
  //let r = (swap (& n       4) r Free)
  //let r = (swap (& n       8) r Free)
  //let r = (swap (& n      16) r Free)
  //let r = (swap (& n      32) r Free)
  //let r = (swap (& n      64) r Free)
  //let r = (swap (& n     128) r Free)
  //let r = (swap (& n     256) r Free)
  //let r = (swap (& n     512) r Free)
  //let r = (swap (& n    1024) r Free)
  //let r = (swap (& n    2048) r Free)
  //let r = (swap (& n    4096) r Free)
  //let r = (swap (& n    8192) r Free)
  //let r = (swap (& n   16384) r Free)
  //let r = (swap (& n   32768) r Free)
  //let r = (swap (& n   65536) r Free)
  //let r = (swap (& n  131072) r Free)
  //let r = (swap (& n  262144) r Free)
  //let r = (swap (& n  524288) r Free)
  //let r = (swap (& n 1048576) r Free)
  //let r = (swap (& n 2097152) r Free)
  //let r = (swap (& n 4194304) r Free)
  //let r = (swap (& n 8388608) r Free)
  //r

//// swap : u32 -> Map -> Map -> Map
//swap = λn match n {
  //0:   λx0 λx1 (Node x0 x1)
  //1+p: λx0 λx1 (Node x1 x0)
//}

//// main : u32
//main = (sum (sort (rev (gen 22 0))))

//TASK: convert the program above to JavaScript. use λ-encodings. keep the style as close as possible


const Empty = empty => single => concat => empty;
const Single = x => empty => single => concat => single(x);
const Concat = x0 => x1 => empty => single => concat => concat(x0)(x1);

const Free = free => used => node => free;
const Used = free => used => node => used;
const Node = x0 => x1 => free => used => node => node(x0)(x1);

const gen = n => x => {
  if (n === 0) {
    return Single(x);
  } else {
    const p  = n - 1;
    const x0 = x << 1;
    const x1 = x0 | 1;
    return Concat(gen(p)(x0))(gen(p)(x1));
  }
};

const rev = a => {
  const a_empty  = Empty;
  const a_single = x => Single(x);
  const a_concat = x0 => x1 => Concat(rev(x1))(rev(x0));
  return a(a_empty)(a_single)(a_concat);
};

const sum = a => {
  const a_empty  = 0;
  const a_single = x => x;
  const a_concat = x0 => x1 => sum(x0) + sum(x1);
  return a(a_empty)(a_single)(a_concat);
};

const sort = t => to_arr(to_map(t))(0);

const to_arr = a => k => {
  const a_free = Empty;
  const a_used = Single(k);
  const a_node = x0 => x1 => {
    const x0Arr = to_arr(x0)(k * 2);
    const x1Arr = to_arr(x1)(k * 2 + 1);
    return Concat(x0Arr)(x1Arr);
  };
  return a(a_free)(a_used)(a_node);
};

const to_map = a => {
  const a_empty  = Free;
  const a_single = x => radix(x);
  const a_concat = x0 => x1 => merge(to_map(x0))(to_map(x1));
  return a(a_empty)(a_single)(a_concat);
};

const merge = a => {
  const a_free = b => {
    const b_free = Free;
    const b_used = Used;
    const b_node = b0 => b1 => Node(b0)(b1);
    return b(b_free)(b_used)(b_node);
  };
  const a_used = b => {
    const b_free = Used;
    const b_used = Used;
    const b_node = b0 => b1 => 0;
    return b(b_free)(b_used)(b_node);
  };
  const a_node = a0 => a1 => b => {
    const b_free = a0 => a1 => Node(a0)(a1);
    const b_used = a0 => a1 => 0;
    const b_node = b0 => b1 => a0 => a1 => Node(merge(a0)(b0))(merge(a1)(b1));
    return b(b_free)(b_used)(b_node)(a0)(a1);
  };
  return a(a_free)(a_used)(a_node);
};

const radix = n => {
  let r = Used;
  r = swap(n & 1)(r)(Free);
  r = swap(n & 2)(r)(Free);
  r = swap(n & 4)(r)(Free);
  r = swap(n & 8)(r)(Free);
  r = swap(n & 16)(r)(Free);
  r = swap(n & 32)(r)(Free);
  r = swap(n & 64)(r)(Free);
  r = swap(n & 128)(r)(Free);
  r = swap(n & 256)(r)(Free);
  r = swap(n & 512)(r)(Free);
  r = swap(n & 1024)(r)(Free);
  r = swap(n & 2048)(r)(Free);
  r = swap(n & 4096)(r)(Free);
  r = swap(n & 8192)(r)(Free);
  r = swap(n & 16384)(r)(Free);
  r = swap(n & 32768)(r)(Free);
  r = swap(n & 65536)(r)(Free);
  r = swap(n & 131072)(r)(Free);
  r = swap(n & 262144)(r)(Free);
  r = swap(n & 524288)(r)(Free);
  r = swap(n & 1048576)(r)(Free);
  r = swap(n & 2097152)(r)(Free);
  r = swap(n & 4194304)(r)(Free);
  r = swap(n & 8388608)(r)(Free);
  return r;
};

const swap = n => x0 => x1 => {
  if (n === 0) {
    return Node(x0)(x1);
  } else {
    return Node(x1)(x0);
  }
};

console.log(sum(sort(rev(gen(22)(0)))));
