#![cfg(feature = "_fuzz")]
#![feature(const_type_name)]

use hvmc::{
  fuzz::*,
  run::{Addr, Heap, Port, Strict, Tag},
  trace,
};

use serial_test::serial;

type Net<'a> = hvmc::run::Net<'a, Strict>;

#[test]
#[serial]
fn fuzz_var_link_link_var() {
  assert!(cfg!(not(feature = "_fuzz_no_free")));
  trace::set_hook();
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let heap = Heap::new(256);
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
    let mut nets = net.fork(2);
    let mut n0 = nets.next().unwrap();
    let mut n1 = nets.next().unwrap();
    fuzz.scope(|s| {
      s.spawn(|| {
        let (x, y) = fuzz.maybe_swap(b.clone(), c.clone());
        n0.link_wire_wire(x.wire(), y.wire());
      });
      s.spawn(|| {
        let (x, y) = fuzz.maybe_swap(d.clone(), e.clone());
        n1.link_wire_wire(x.wire(), y.wire());
      });
    });
    let used = assert_linked(a, f);
    for x in [b, c, d, e] {
      if !used.contains(&x.addr()) && x.addr().val().read() != Port::FREE.0 {
        panic!("failed to free");
      }
    }
  });
}

#[test]
#[serial]
fn fuzz_pri_link_link_pri() {
  assert!(cfg!(not(feature = "_fuzz_no_free")));
  trace::set_hook();
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let p = Port::new(Tag::Ctr, 0, Addr::NULL);
    let q = Port::new(Tag::Ctr, 1, Addr::NULL);
    let heap = Heap::new(256);
    let mut net = Net::new(&heap);
    let x = net.alloc();
    let a = Port::new_var(x.clone());
    let b = Port::new_var(x.other_half());
    net.link_port_port(a.clone(), b.clone());
    let mut nets = net.fork(2);
    let mut n0 = nets.next().unwrap();
    let mut n1 = nets.next().unwrap();
    fuzz.scope(|s| {
      s.spawn(|| {
        n0.link_wire_port(a.wire(), p);
      });
      s.spawn(|| {
        n1.link_wire_port(b.wire(), q);
      });
    });
    assert!(n0.redexes.len() == 1 || n1.redexes.len() == 1);
    for x in [a, b] {
      assert_eq!(x.addr().val().read(), Port::FREE.0);
    }
  })
}

#[test]
#[serial]
fn fuzz_var_link_link_pri() {
  assert!(cfg!(not(feature = "_fuzz_no_free")));
  trace::set_hook();
  let heap = Heap::new(256);
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let mut net = Net::new(&heap);
    let x = net.alloc();
    let y = net.alloc();
    let a = Port::new_var(x.clone());
    let b = Port::new_var(x.other_half());
    let c = Port::new_var(y.clone());
    let d = Port::new_var(y.other_half());
    net.link_port_port(a.clone(), b.clone());
    net.link_port_port(c.clone(), d.clone());
    let mut nets = net.fork(2);
    let mut n0 = nets.next().unwrap();
    let mut n1 = nets.next().unwrap();
    fuzz.scope(|s| {
      s.spawn(|| {
        let (x, y) = fuzz.maybe_swap(b.clone(), c.clone());
        n0.link_wire_wire(x.wire(), y.wire());
      });
      s.spawn(|| {
        n1.link_wire_port(d.wire(), Port::ERA);
      });
    });
    let at = Port(a.addr().val().read());
    assert_eq!(at, Port::ERA);
    // TODO: reenable leak detection
    if false {
      for x in [b, c, d] {
        assert_eq!(Port(x.addr().val().read()), Port::FREE, "failed to free {:?}", x.wire());
      }
    }
  })
}

#[test]
#[serial]
#[ignore = "very slow"] // takes ~50m on my M3 Max (or ~13.5h with tracing enabled)
fn fuzz_var_link_link_link_var() {
  assert!(cfg!(feature = "_fuzz_no_free"));
  trace::set_hook();
  let heap = Heap::new(256);
  Fuzzer::default().fuzz(|fuzz| {
    unsafe { trace::_reset_traces() };
    let mut net = Net::new(&heap);
    let x = net.alloc();
    let y = net.alloc();
    let z = net.alloc();
    let w = net.alloc();
    let a = Port::new_var(x.clone());
    let b = Port::new_var(x.other_half());
    let c = Port::new_var(y.clone());
    let d = Port::new_var(y.other_half());
    let e = Port::new_var(z.clone());
    let f = Port::new_var(z.other_half());
    let g = Port::new_var(w.clone());
    let h = Port::new_var(w.other_half());
    net.link_port_port(a.clone(), b.clone());
    net.link_port_port(c.clone(), d.clone());
    net.link_port_port(e.clone(), f.clone());
    net.link_port_port(g.clone(), h.clone());
    let mut nets = net.fork(3);
    let mut n0 = nets.next().unwrap();
    let mut n1 = nets.next().unwrap();
    let mut n2 = nets.next().unwrap();
    fuzz.scope(|s| {
      s.spawn(|| {
        let (x, y) = fuzz.maybe_swap(b.clone(), c.clone());
        n0.link_wire_wire(x.wire(), y.wire());
      });
      s.spawn(|| {
        let (x, y) = fuzz.maybe_swap(d, e);
        n1.link_wire_wire(x.wire(), y.wire());
      });
      s.spawn(|| {
        let (x, y) = fuzz.maybe_swap(f, g);
        n2.link_wire_wire(x.wire(), y.wire());
      });
    });
    assert_linked(a, h);
  })
}

fn assert_linked(x: Port, y: Port) -> Vec<Addr> {
  let mut used = vec![];
  for (x, y) in [(x.clone(), y.clone()), (y, x)] {
    let mut p = Port(x.wire().addr().val().read());
    if p != y {
      loop {
        used.push(p.addr());
        let q = Port(p.addr().val().read());
        if p.tag() == Tag::Red && p.wire() == y.wire() {
          break;
        }
        if q.tag() == Tag::Red {
          p = q;
          continue;
        }
        panic!("bad link");
      }
    }
  }
  used
}
