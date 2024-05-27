C2 = λfλx(f (f x))
C6 = λfλx(f (f (f (f (f (f x))))))
C1 = λfλx(f x)
C4 = λfλx(f (f (f (f x))))
C3 = λfλx(f (f (f x)))
plus = λmλnλfλx(m f (n f x))
fib = 
	λn
	let (*, r) = (n λx(let (a, b) = x; (b, (plus a b))) (C1, C1))
	r

expensive = (fib (C4 C2))

erase = λn λx (n HVM.black_box x)
black_plus = (HVM.black_box plus)
expensive_1 = (fib (C4 C2))
expensive_2 = (fib (C4 C2))

main_fast = (black_plus expensive   expensive   )
main_slow = (black_plus expensive_1 expensive_2 )
main = *