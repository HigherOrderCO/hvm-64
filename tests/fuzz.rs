#![cfg(feature = "_fuzz")]

use std::collections::HashSet;

use hvmc::{
  fuzz::*,
  run::{Loc, Net, Port, Tag},
  trace,
};

use serial_test::serial;

#[test]
fn fuzz_xy() {
  let mut results = HashSet::new();
  Fuzzer::default().fuzz(|f| {
    let x = AtomicU64::new(0);
    let y = AtomicU64::new(1);
    f.scope(|s| {
      s.spawn(|| {
        y.store(3, Ordering::Relaxed);
        x.store(1, Ordering::Relaxed);
      });
      s.spawn(|| {
        if x.load(Ordering::Relaxed) == 1 {
          y.store(y.load(Ordering::Relaxed) * 2, Ordering::Relaxed);
        }
      });
    });
    results.insert(y.read());
  });
  assert_eq!(results, [6, 3, 2].into_iter().collect());
}

#[test]
#[serial]
fn fuzz_var_link_link_var() {
  trace::set_hook();
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let heap = Net::init_heap(256);
    let mut net = Net::new(&heap);
    let x = net.alloc();
    let y = net.alloc();
    let z = net.alloc();
    let a = Port::new_var(x.clone());
    let b = Port::new_var(x.other_half());
    let c = Port::new_var(y.clone());
    let d = Port::new_var(y.other_half());
    let e = Port::new_var(z.clone());
    let f = Port::new_var(z.other_half());
    net.link_port_port(a.clone(), b.clone());
    net.link_port_port(c.clone(), d.clone());
    net.link_port_port(e.clone(), f.clone());
    let mut n0 = net.fork(0, 2);
    let mut n1 = net.fork(1, 2);
    fuzz.scope(move |s| {
      s.spawn(move || {
        let (x, y) = fuzz.maybe_swap(b, c);
        n0.link_wire_wire(x.wire(), y.wire());
      });
      s.spawn(move || {
        let (x, y) = fuzz.maybe_swap(d, e);
        n1.link_wire_wire(x.wire(), y.wire());
      });
    });
    let at = Port(a.loc().val().read());
    let ft = Port(f.loc().val().read());
    // dbg!(&a, &f, &at, &ft);
    if at != f || ft != a {
      panic!("invalid link")
    }
  })
}

#[test]
#[serial]
fn fuzz_pri_link_link_pri() {
  trace::set_hook();
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let p = Port::new(Tag::Ctr, 0, Loc::NULL);
    let q = Port::new(Tag::Ctr, 1, Loc::NULL);
    let heap = Net::init_heap(256);
    let mut net = Net::new(&heap);
    let x = net.alloc();
    let a = Port::new_var(x.clone());
    let b = Port::new_var(x.other_half());
    net.link_port_port(a.clone(), b.clone());
    let mut n0 = net.fork(0, 2);
    let mut n1 = net.fork(1, 2);
    fuzz.scope(|s| {
      s.spawn(|| {
        n0.link_wire_port(a.wire(), p);
      });
      s.spawn(|| {
        n1.link_wire_port(b.wire(), q);
      });
    });
    assert!(n0.rdex.len() == 1 || n1.rdex.len() == 1);
  })
}

#[test]
#[serial]
fn fuzz_var_link_link_pri() {
  trace::set_hook();
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let heap = Net::init_heap(256);
    let mut net = Net::new(&heap);
    let x = net.alloc();
    let y = net.alloc();
    let a = Port::new_var(x.clone());
    let b = Port::new_var(x.other_half());
    let c = Port::new_var(y.clone());
    let d = Port::new_var(y.other_half());
    net.link_port_port(a.clone(), b.clone());
    net.link_port_port(c.clone(), d.clone());
    let mut n0 = net.fork(0, 2);
    let mut n1 = net.fork(1, 2);
    fuzz.scope(move |s| {
      s.spawn(move || {
        let (x, y) = fuzz.maybe_swap(b, c);
        n0.link_wire_wire(x.wire(), y.wire());
      });
      s.spawn(move || {
        n1.link_wire_port(d.wire(), Port::ERA);
      });
    });
    let at = Port(a.loc().val().read());
    assert_eq!(at, Port::ERA);
  })
}
