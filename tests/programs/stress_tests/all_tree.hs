data Bool = True | False
data Tree = Node Tree Tree | Leaf Bool

and' :: Bool -> Bool -> Bool
and' a b = case a of
  True -> b
  False -> False

gen :: Int -> Tree
gen n = case n of
  0 -> Leaf True
  p -> let p' = p - 1 in Node (gen p') (gen p')

all' :: Tree -> Bool
all' t = case t of
  Node a b -> and' (all' a) (all' b)
  Leaf v -> v

main :: Bool
main = all' (gen 22)

