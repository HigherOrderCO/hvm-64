// A cool trick involving HVM's scopeless lambdas is linear queues:

// Queue.new : Queue a
Queue.new = λx x

// Queue.add : a -> Queue a -> Queue a
Queue.add = λx λq λk (q λc (c x k))

// Queue.rem : Queue a -> Pair a (Queue a)
Queue.rem = λq (q $k λx λxs λp(p x λ$k xs))

Nil  = λc λn n
Cons = λh λt λc λn (c h t)

// Church Nats
C0 = λs λz z
C1 = λs λz (s z)
C2 = λs λz (s (s z))

// Output: [1, 2, 3]
main =
  let q = Queue.new
  let q = (Queue.add C0 q)
  let q = (Queue.add C1 q)
  let q = (Queue.add C2 q)
  (Queue.rem q λv0 λq
  (Queue.rem q λv1 λq
  (Queue.rem q λv2 λq
  (Cons C0 (Cons C1 (Cons C2 Nil))))))