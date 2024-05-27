S = λn λs λz (s (n s z))
Z = λs λz z

c2 = λf λx (f (f x))
c3 = λf λx (f (f (f x)))
c4 = (S (S (S (S Z))))

add = λa λb λs λz (a s (b s z))
mul = λa λb λs λz (a (b s) z)

// 2 * 3 + 4
main = (add (mul c2 c3) c4)
