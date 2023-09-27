HVM2 WIP
========

## This is a work-in-progress.

Explanation from Discord:

> HVM2 is an attempt to greatly simplify HVM, while retaining (or, hopefully, improving) its performance. We want to remove constructors and pattern-match in favor of just interaction combinators. This greatly simplifies the implementation, but could affect performance. To regain the performance, we 1. improve memory footprint of interaction combinators by using HVM-like layout, 2. implement supercombinator-based definitions, which allow for lightweight recursive functions, 3. implementing it on GPUs, which is possible with a new lock-free reduction algorithm. So, in short, HVM2 is a major simplification of HVM1, which we're trying to make as fast as (or faster). If it works out, HVM2 will completely replace HVM1 and will be launched as a seamless update, with backwards compatibility. I must note that HVM1 is a prototype due to parallelism bugs and edge cases, that HVM2 completely solves. I'll release a paper explaining the new lock-free parallel reduction algorithm on the next days.
