sum = λa match a {
  0: λs s
  1+: λs (sum a-1 (+ a-1 s))
}

main = (sum 10000000 0)
