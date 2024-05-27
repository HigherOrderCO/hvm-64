use super::*;

use hvm64_util::maybe_grow;

/// Calculates the labels used in each definition of a book.
///
/// # Background: Naive Algorithms
///
/// The simplest algorithm to calculate labels would be to go to each def,
/// recursively traverse the tree (going into references), and collect all of
/// the labels.
///
/// Now, this algorithm will not terminate for recursive definitions, but fixing
/// this is relatively simple: don't enter a reference twice in one traversal.
///
/// This modified algorithm will work in all cases, but it's not very efficient.
/// Notably, it can take quadratic time; for example, it requires traversing
/// each of these refs 3 times:
/// ```text
/// @foo = (* @bar)
/// @bar = (* @baz)
/// @baz = (* @foo)
/// ```
///
/// This can be resolved with memoization. However, a simple memoization pass
/// will not work here, due to the cycle avoidance algorithm. For example,
/// consider the following program:
/// ```text
/// @foo = (@bar {1 x x})
/// @bar = ({2 x x} @foo)
/// ```
///
/// A simple memoization pass would go as follows:
/// - calculate the labels for `@foo`
///   - encounter `@bar`
///     - this has not yet been processed, so calculate the labels for it
///       - encounter `{2 ...}`; add 2 to the label set
///       - encounter `@foo` -- this has already been visited, so skip it (cycle
///         avoidance)
///     - add all the labels from `@bar` (just 2) to `@foo`
///   - encounter `{1 ...}`; add 1 to the label set
/// - calculate the labels for `@bar` -- this has already been visited, so skip
///   it (memoization)
///
/// The end result of this is `@foo: {1, 2}, @bar: {2}` -- `@bar` is missing
/// `1`.
///
/// Instead, a two-phase memoization approach is needed.
///
/// # Implemented Here: Two-Phase Algorithm
///
/// Each phase follows a similar procedure as the simple memoization algorithm,
/// with a few modifications. (Henceforth, the *head node* of a cycle is the
/// first node in a cycle reached in a traversal.)
/// - in the first phase:
///   - when we process a node involved in a cycle:
///     - if it is not the head node of a cycle it is involved in, instead of
///       saving the computed label set to the memo, we store a placeholder
///       `Cycle` value into the memo
///     - otherwise, after storing the label set in the memo, we enter the
///       second phase, retraversing the node
///   - when we encounter a `Cycle` value in the memo, we treat it akin to an
///     empty label set
/// - in the second phase, we treat `Cycle` values as though they were missing
///   entries in the memo
///
/// In the first phase, to know when we are processing nodes in a cycle, we keep
/// track of the depth of the traversal, and every time we enter a ref, we store
/// `Cycle(depth)` into the memo -- so if we encounter this in the traversal of
/// the children, we know that that node is involved in a cycle. Additionally,
/// when we exit a node, we return the *head depth* of the result that was
/// calculated -- the least depth of a node involved in a cycle with the node
/// just processed. Comparing this with the depth gives us the information
/// needed for the special rules.
///
/// This algorithm runs in linear time (as refs are traversed at most twice),
/// and requires no more space than the naive algorithm.
pub(crate) fn calculate_label_sets<'b, 'l>(book: &'b Book, lookup: impl FnMut(&'b str) -> LabSet) -> LabelSets<'b> {
  let mut state = State {
    book,
    lookup,
    #[cfg(feature = "std")]
    labels: Map::with_capacity(book.len()),
    #[cfg(not(feature = "std"))]
    labels: Map::new(),
  };

  for name in book.keys() {
    state.visit_def(name, Some(0), None);
  }

  LabelSets(state.labels)
}

pub(crate) struct LabelSets<'b>(Map<&'b str, LabelState>);

impl<'b> LabelSets<'b> {
  pub(crate) fn into_iter(self) -> impl Iterator<Item = (&'b str, LabSet)> {
    self.0.into_iter().map(|(nam, lab)| match lab {
      LabelState::Done(lab) => (nam, lab),
      _ => unreachable!(),
    })
  }
}

struct State<'b, F> {
  book: &'b Book,
  lookup: F,
  labels: Map<&'b str, LabelState>,
}

#[derive(Debug)]
enum LabelState {
  Done(LabSet),
  /// Encountering this node indicates participation in a cycle with the given
  /// head depth.
  Cycle(usize),
}

/// All of these methods share a similar signature:
/// - `depth` is optional; `None` indicates that this is the second processing
///   pass (where the depth is irrelevant, as all cycles have been detected)
/// - `out`, if supplied, will be unioned with the result of this traversal
/// - the return value indicates the head depth, as defined above (or an
///   arbitrary value `>= depth` if no cycles are involved)
impl<'b, F: FnMut(&'b str) -> LabSet> State<'b, F> {
  fn visit_def(&mut self, key: &'b str, depth: Option<usize>, out: Option<&mut LabSet>) -> usize {
    match self.labels.entry(key) {
      Entry::Vacant(e) => {
        e.insert(LabelState::Cycle(depth.unwrap()));
        self.calc_def(key, depth, out)
      }
      Entry::Occupied(mut e) => match e.get_mut() {
        LabelState::Done(ref labs) => {
          if let Some(out) = out {
            out.union(labs);
          }
          usize::MAX
        }
        LabelState::Cycle(d) if depth.is_some() => *d,
        LabelState::Cycle(_) => {
          e.insert(LabelState::Done(LabSet::default()));
          self.calc_def(key, depth, out)
        }
      },
    }
  }

  fn calc_def(&mut self, key: &'b str, depth: Option<usize>, out: Option<&mut LabSet>) -> usize {
    let mut labs = LabSet::default();
    let head_depth = self.visit_within_def(key, depth, Some(&mut labs));
    if let Some(out) = out {
      out.union(&labs);
    }
    if depth.is_some_and(|x| x > head_depth) {
      self.labels.insert(key, LabelState::Cycle(head_depth));
    } else {
      self.labels.insert(key, LabelState::Done(labs));
      if depth == Some(head_depth) {
        self.visit_within_def(key, None, None);
      }
    }
    head_depth
  }

  fn visit_within_def(&mut self, key: &str, depth: Option<usize>, mut out: Option<&mut LabSet>) -> usize {
    let def = &self.book[key];
    let mut head_depth = self.visit_tree(&def.root, depth, out.as_deref_mut());
    for (a, b) in &def.redexes {
      head_depth = head_depth.min(self.visit_tree(a, depth, out.as_deref_mut()));
      head_depth = head_depth.min(self.visit_tree(b, depth, out.as_deref_mut()));
    }
    head_depth
  }

  fn visit_tree(&mut self, tree: &'b Tree, depth: Option<usize>, mut out: Option<&mut LabSet>) -> usize {
    maybe_grow(move || {
      if let Some(lab) = tree.lab() {
        if let Some(out) = out.as_deref_mut() {
          out.add(lab);
        }
      }
      if let Tree::Ref { nam } = tree {
        if self.book.contains_key(nam) {
          return self.visit_def(nam, depth.map(|x| x + 1), out);
        }
        if let Some(out) = &mut out {
          out.union(&(self.lookup)(nam));
        }
      }
      tree.children().map(|child| self.visit_tree(child, depth, out.as_deref_mut())).fold(usize::MAX, usize::min)
    })
  }
}

#[test]
fn test_calculate_labels() {
  use alloc::collections::BTreeMap;
  assert_eq!(
    calculate_label_sets(
      &"
        @a = {0 @b @c}
        @b = {1 @a *}
        @c = {2 @a *}

        @p = {3 @q {4 x x}}
        @q = {5 @r @t}
        @r = {6 @s *}
        @s = {7 @r *}
        @t = {8 @q {9 @t @u}}
        @u = {10 @u @s}
      "
      .parse()
      .unwrap(),
      |_| unreachable!(),
    )
    .into_iter()
    .collect::<BTreeMap<_, _>>(),
    [
      ("a", [0, 1, 2].into_iter().collect()),
      ("b", [0, 1, 2].into_iter().collect()),
      ("c", [0, 1, 2].into_iter().collect()),
      //
      ("p", [3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect()),
      ("q", [5, 6, 7, 8, 9, 10].into_iter().collect()),
      ("r", [6, 7].into_iter().collect()),
      ("s", [6, 7].into_iter().collect()),
      ("t", [5, 6, 7, 8, 9, 10].into_iter().collect()),
      ("u", [6, 7, 10].into_iter().collect()),
    ]
    .into_iter()
    .collect()
  );
}
