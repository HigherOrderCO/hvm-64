//node = λa λb λnode λleaf (node a b)
//leaf = λv    λnode λleaf (leaf v)

//gen = λn match n {
  //0   : (leaf 1)
  //1+p : (node (gen p) (gen p))
//}

//sum = λt
  //let case_node = λa λb (+ (sum a) (sum b))
  //let case_leaf = λv v
  //(t case_node case_leaf)

//main = (sum (gen 24))

const node = a => b => nodeFn => leafFn => nodeFn(a)(b);
const leaf = v => nodeFn => leafFn => leafFn(v);

const gen = n => n == 0 ? leaf(1) : node(gen(n-1))(gen(n-1));

const sum = t => {
  const case_node = a => b => sum(a) + sum(b);
  const case_leaf = v => v;
  return t(case_node)(case_leaf);
};

console.log(sum(gen(24)));
