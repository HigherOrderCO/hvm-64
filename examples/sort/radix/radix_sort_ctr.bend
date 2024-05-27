data Arr = Empty | (Single x) | (Concat x0 x1)

data Map = Free | Used | (Node x0 x1)

// gen : u32 -> Arr
gen = λn match n {
  0: λx (Single x)
  1+: λx
    let x0 = (<< x 1)
    let x1 = (| x0 1)
    (Concat (gen n-1 x0) (gen n-1 x1))
}

// rev : Arr -> Arr
rev Empty = Empty
rev (Single x) = (Single x)
rev (Concat a b) = (Concat (rev b) (rev a))

// sum : Arr -> u32
sum Empty = 0
sum (Single x) = x
sum (Concat a b) = (+ (sum a) (sum b))

// sort : Arr -> Arr
sort = λt (to_arr (to_map t) 0)

// to_arr : Map -> u32 -> Arr
to_arr Free * = Empty
to_arr Used k = (Single k)
to_arr (Node a b) k = (Concat (to_arr a (* k 2)) (to_arr b (+ (* k 2) 1)))

// to_map : Arr -> Map
to_map Empty = Free
to_map (Single x) = (radix x)
to_map (Concat a b) = (merge (to_map a) (to_map b))

// merge : Map -> Map -> Map
merge Free x = x
merge Used Free = Used
merge Used Used = Used
merge Used (Node * *) = *
merge (Node a b) Free = (Node a b)
merge (Node * *) Used = *
merge (Node a b) (Node c d) = (Node (merge a c) (merge b d))

// radix : u32 -> Map
radix = λn
  let r = Used
  let r = (swap (& n       1) r Free)
  let r = (swap (& n       2) r Free)
  let r = (swap (& n       4) r Free)
  let r = (swap (& n       8) r Free)
  let r = (swap (& n      16) r Free)
  let r = (swap (& n      32) r Free)
  let r = (swap (& n      64) r Free)
  let r = (swap (& n     128) r Free)
  let r = (swap (& n     256) r Free)
  let r = (swap (& n     512) r Free)
  let r = (swap (& n    1024) r Free)
  let r = (swap (& n    2048) r Free)
  let r = (swap (& n    4096) r Free)
  let r = (swap (& n    8192) r Free)
  let r = (swap (& n   16384) r Free)
  let r = (swap (& n   32768) r Free)
  let r = (swap (& n   65536) r Free)
  let r = (swap (& n  131072) r Free)
  let r = (swap (& n  262144) r Free)
  let r = (swap (& n  524288) r Free)
  let r = (swap (& n 1048576) r Free)
  let r = (swap (& n 2097152) r Free)
  let r = (swap (& n 4194304) r Free)
  let r = (swap (& n 8388608) r Free)
  r

// swap : u32 -> Map -> Map -> Map
swap = λn match n {
  0: λx0 λx1 (Node x0 x1)
  1+: λx0 λx1 (Node x1 x0)
}

// main : u32
main = (sum (sort (rev (gen 20 0))))
