// This is, as far as I can tell, the most efficient solution to the atomic
// substitution problem: https://www.youtube.com/watch?v=PCAfVdmBYXQ
//
// It is an hybrid of the locking and lock-free approach that will try to
// perform the link with 2 atomic CAS. For each CAS that fails, we place a
// redirection node on the corresponding port. We then start a soft 'link(x)'
// procedure that will *try* to remove redirections. If 'x' is an aux port, we
// do not own it, so, some redirections may remain. If 'x' is a main port (not
// implemented here), we own it, so we will successfully remove all redirections
// after it, and either insert it in an aux port, or find a new active pair.
// This algorithm is superior because, in most cases, the 2 initial CAS will
// succeed, allowing us to avoid storing redirections and calling 'link()'. Both
// Oliver, Derenash and I independently concluded this is the best solution.

// Atomic Primitives
// -----------------

function* atomic_read(idx) {
  yield null;
  return D[idx];
}

function* atomic_swap(idx, val) {
  yield null;
  let got = D[idx];
  D[idx] = val;
  return got;
}

function* atomic_cas(idx, from, to) {
  yield null;
  let old = D[idx];
  if (old === from) {
    D[idx] = to;
  }
  return old;
}

// Atomic Substitution
// -------------------

const RED = 0x10; // redirection wire
const GOT = 0xFF; // "taken" placeholder

const is_red = (x) => x >= RED && x < RED*2;
const is_var = (x) => x >= 0 && x < RED;

function* atomic_subst(tid, a_dir, b_dir) {
  var a_ptr = yield* atomic_swap(a_dir, GOT);
  var b_ptr = yield* atomic_swap(b_dir, GOT);
  var a_got = yield* atomic_cas(a_ptr, a_dir, b_ptr);
  var b_got = yield* atomic_cas(b_ptr, b_dir, a_ptr);
  yield* atomic_swap(a_dir, a_got == a_dir ? -1 : b_ptr+RED);
  yield* atomic_swap(b_dir, b_got == b_dir ? -1 : a_ptr+RED);
  yield* atomic_link(tid, a_ptr);
  yield* atomic_link(tid, b_ptr);
}

function* atomic_link(tid, dir) {
  var ptr = yield* atomic_read(dir);
  if (is_var(ptr)) {
    while (1) {
      var trg = yield* atomic_read(ptr);
      if (is_red(trg)) {
        var got = yield* atomic_cas(dir, ptr, trg-RED);
        if (got == ptr) {
          yield* atomic_swap(ptr, -1);
          ptr = trg - RED;
          continue;
        }
      }
      break;
    }
  }
}

// Utils
// -----

function progress(ticks) {
  for (var i = 0; i < ticks; ++i) {
    var tid = Math.floor(Math.random() * 3);
    if (T[tid]) {
      let result = T[tid].next();
      if (result.done) {
        T[tid] = null;
      }
      if (!T[0] && !T[1] && !T[2]) {
        break;
      }
    }
  }
}

function link(dir) {
  var t = atomic_link(0, dir);
  while (!t.next().done) {};
}

function show(x) {
  if (typeof x == "number") {
    return x == -1 ? "  " : x == 0xFF ? "::" : ("00"+x.toString(16)).slice(-2);
  } else {
    return x.map(show).join("|");
  }
}

// Tests
// -----

var I = [0,1,2,3,4,5,6,7];
var D = [1,0,3,2,5,4,7,6];
var T = [atomic_subst(0,1,2), atomic_subst(1,3,4), atomic_subst(2,5,6)];
progress(256);

console.log(show(I));
console.log(show(D));
D.map((x,i) => link(i));
console.log(show(D));
console.log(T);
