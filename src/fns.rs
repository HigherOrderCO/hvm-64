use crate::run::{*};
pub const ADD: Val = 0x249e7; // 'add'

impl<'a> Net<'a> {
  // recursively walk down the args list, saving each native arg to a vector
  // returns a tuple of the args vector and a pointer to location for result
  fn extract_args(&mut self, book: &Book, args: Ptr, arg_count: usize) -> (Vec<Ptr>, Ptr) {
    let args_loc = args.loc();

    // if the argument position has a LAM node
    if args.tag() >= LAM {
      // P1 points to the arg
      // peek the value at args_loc P1
      let arg = self.heap.get(args_loc, P1);
      if arg.tag() == NUM {
        // peek the value at args_loc P2
        // P2 points to the rest of the args list
        let rest_args = self.heap.swap(args_loc, P2, LOCK);
        if rest_args == LOCK {
          // somebody else owns where the result needs to go,
          // so stop getting args and let the caller know where
          // we stopped looking so the proper steps can be taken
          return (vec![], args);
        } else if arg_count == 1 {
          // we have all the args we were told to get,
          // return the args list and the output pointer
          self.heap.set(args_loc, P2, ERAS);
          return (vec![arg], rest_args);
        } else {
          // restore rest_args
          self.heap.swap(args_loc, P2, rest_args);

          // recurse to get the rest of the args
          let (mut args, out) = self.extract_args(book, rest_args, arg_count - 1);

          // add the current arg to end of the 'args' vector
          args.push(arg);
          return (args, out);
        }
      } else {
        // this argument isn't a native value, so stop getting args
        // and let the caller know where we stopped looking
        return (vec![], args);
      }
    } else {
      // the rest of the args list isn't there, so return what we have
      return (vec![], args);
    }
  }

  // extract the requested number of native args
  fn native_args(&mut self, book: &Book, ptr: Ptr, args: Ptr, arg_count: usize)
     -> Option<(Vec<Ptr>, Ptr)> {
    // try to get the correct number of arguments
    let (native_args, out) = self.extract_args(book, args, arg_count);

    if native_args.len() == arg_count {
      // we got all the args requested, so return them and
      // the ptr to where to put the results

      // before we go, free the args list
      self.redux(args, ERAS);
      return Some((native_args, out));
    } else {
      // 'out' was already LOCKED by us
      match out.tag() {
        VR1 | VR2 => {
          // we need to wait on the rest of the args list
          let rdx = self.alloc();
          self.heap.set(rdx, P1, ptr);
          self.heap.set(rdx, P2, args);

          let new_out = self.get_target(out);
          let out_loc = self.swap_target(new_out, Ptr::new(RDX, 0, rdx));
          self.set_target(out_loc, out);
        }
        LAM.. =>  {
          // we didn't get enough native args, so further processing is needed
          // 'out' is the node that points to the non-NUM arg with P1
          let bad_arg = self.heap.swap(out.loc(), P1, LOCK);
          match bad_arg.tag() {
            0xF => {
              // someone is trying to supply an arg, so just retry the call
              self.redux(ptr, args);
            }
            ERA => self.redux(args, ERAS),
            REF => {
              // load the ref's closed net
              let got = book.get(bad_arg.val()).unwrap();
              let b = self.load_net(got);

              // update the arg
              self.heap.set(out.loc(), P1, b);

              // try the native call again
              self.redux(ptr, args);
            }
            VR1 | VR2 => {
              // the arg hasn't been supplied yet
              // so store the redex on the heap
              let rdx = self.alloc();
              self.heap.set(rdx, P1, ptr);
              self.heap.set(rdx, P2, args);

              // and wait for it
              self.heap.set(out.loc(), P1, Ptr::new(RDX, 0, rdx));
            }
            LAM.. => {
              // multiple args are supplied for this arg position
              // so duplicate the native call across the args for this position
              let arg1 = self.heap.swap(bad_arg.loc(), P1, LOCK);
              let arg2 = self.heap.swap(bad_arg.loc(), P2, LOCK);
              let new_out = self.heap.swap(out.loc(), P2, LOCK);
              if arg1 == LOCK || arg2 == LOCK || new_out == LOCK {
                // someone is trying to supply us needed values
                // we don't care if the CAS fails, that just means they've already
                // written the value we were waiting on.
                let _ = self.heap.cas(bad_arg.loc(), P1, LOCK, arg1);
                let _ = self.heap.cas(bad_arg.loc(), P2, LOCK, arg2);
                let _ = self.heap.cas(out.loc(), P2, LOCK, new_out);

                // and just retry
                self.redux(ptr, args);
              }

              // c1 and c2 are the locations for the individual args for this position
              let c1 = self.alloc();
              let c2 = self.alloc();

              // 'results' is where the results from the duplicated calls are gathered
              let results = self.alloc();
              self.heap.set(c1, P2, Ptr::new(VR1, 0, results));
              self.heap.set(c2, P2, Ptr::new(VR2, 0, results));
              self.heap.set(results, P1, Ptr::new(VR2, 0, c1));
              self.heap.set(results, P2, Ptr::new(VR2, 0, c2));

              // start building the duplicated args lists
              let mut args1 = c1;
              let mut args2 = c2;

              let mut i = 0;
              while i < native_args.len() {
                // allocate a new LAM node for each arg position
                let new_args1 = self.alloc();
                let new_args2 = self.alloc();
           
                self.heap.set(new_args1, P1, native_args[i]);
                self.heap.set(new_args1, P2, Ptr::new(LAM, 0, args1));
                self.heap.set(new_args2, P1, native_args[i]);
                self.heap.set(new_args2, P2, Ptr::new(LAM, 0, args2));

                args1 = new_args1;
                args2 = new_args2;
                i = i + 1;
              }

              if new_out.tag() <= VR2 {
                self.link(new_out, Ptr::new(LAM, 0, results));
              } else {
                let rdx_dir = Ptr::new(LAM, bad_arg.lab(), results);
                self.comm(new_out, rdx_dir);
              }

              // TODO: always just call redux
              if arg1.tag() <= VR2 {
                let rdx = self.alloc();
                self.heap.set(rdx, P1, ptr);
                self.heap.set(rdx, P2, Ptr::new(LAM, 0, args1));
                self.heap.set(c1, P1, Ptr::new(RDX, 0, rdx));
                self.set_target(arg1, Ptr::new(VR1, 0, c1));
              } else {
                self.heap.set(c1, P1, arg1);
                self.redux(ptr, Ptr::new(LAM, 0, args1));
              }

              if arg2.tag() <= VR2 {
                let rdx = self.alloc();
                self.heap.set(rdx, P1, ptr);
                self.heap.set(rdx, P2, Ptr::new(LAM, 0, args2));
                self.heap.set(c2, P1, Ptr::new(RDX, 0, rdx));
                self.set_target(arg2, Ptr::new(VR1, 0, c2));
              } else {
                self.heap.set(c2, P1, arg2);
                self.redux(ptr, Ptr::new(LAM, 0, args2));
              }

              // free the old args list
              self.heap.set(bad_arg.loc(), P1, ERAS);
              self.heap.set(bad_arg.loc(), P2, ERAS);
              self.heap.set(out.loc(), P2, ERAS);
              self.heap.swap(out.loc(), P1, bad_arg);
              self.redux(args, ERAS);
            }
            _ => unreachable!(),
          }
        }
        _ => {
          dbg!(out.tag());
          unreachable!();
        }
      }
    }
    return None;
  }

  fn plus(&mut self, book: &Book, ptr: Ptr, args_list: Ptr) {
    match self.native_args(book, ptr, args_list, 2) {
      None => (),
      Some((args, out)) => {
        let result = (args[1].0 >> 4) + (args[0].0 >> 4);
        let r_ptr = Ptr::big(NUM, result);

        self.link(out, r_ptr);
      }
    }
  }

  pub fn call_native(&mut self, book: &Book, ptr: Ptr, x: Ptr) -> bool {
    match ptr.val() {
      ADD => {
        self.plus(book, ptr, x);
        return true;
      }
      _ => {
        return false;
      }
    }
  }
}
