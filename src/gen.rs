use crate::{
  ast::{DefRef, Host},
  jit::*,
  ops::Op::*,
  run::{Tag::*, *},
};

pub fn host() -> Host {
  let mut host = Host::default();
  host.defs.insert(r#"E"#.to_owned(), DefRef::Static(&DEF_E));
  host.back.insert(Ptr::new_ref(&DEF_E).loc(), r#"E"#.to_owned());
  host.defs.insert(r#"I"#.to_owned(), DefRef::Static(&DEF_I));
  host.back.insert(Ptr::new_ref(&DEF_I).loc(), r#"I"#.to_owned());
  host.defs.insert(r#"O"#.to_owned(), DefRef::Static(&DEF_O));
  host.back.insert(Ptr::new_ref(&DEF_O).loc(), r#"O"#.to_owned());
  host.defs.insert(r#"S"#.to_owned(), DefRef::Static(&DEF_S));
  host.back.insert(Ptr::new_ref(&DEF_S).loc(), r#"S"#.to_owned());
  host.defs.insert(r#"Z"#.to_owned(), DefRef::Static(&DEF_Z));
  host.back.insert(Ptr::new_ref(&DEF_Z).loc(), r#"Z"#.to_owned());
  host.defs.insert(r#"brn"#.to_owned(), DefRef::Static(&DEF_brn));
  host.back.insert(Ptr::new_ref(&DEF_brn).loc(), r#"brn"#.to_owned());
  host.defs.insert(r#"brnS"#.to_owned(), DefRef::Static(&DEF_brnS));
  host.back.insert(Ptr::new_ref(&DEF_brnS).loc(), r#"brnS"#.to_owned());
  host.defs.insert(r#"brnZ"#.to_owned(), DefRef::Static(&DEF_brnZ));
  host.back.insert(Ptr::new_ref(&DEF_brnZ).loc(), r#"brnZ"#.to_owned());
  host.defs.insert(r#"c10"#.to_owned(), DefRef::Static(&DEF_c10));
  host.back.insert(Ptr::new_ref(&DEF_c10).loc(), r#"c10"#.to_owned());
  host.defs.insert(r#"c12"#.to_owned(), DefRef::Static(&DEF_c12));
  host.back.insert(Ptr::new_ref(&DEF_c12).loc(), r#"c12"#.to_owned());
  host.defs.insert(r#"c14"#.to_owned(), DefRef::Static(&DEF_c14));
  host.back.insert(Ptr::new_ref(&DEF_c14).loc(), r#"c14"#.to_owned());
  host.defs.insert(r#"c16"#.to_owned(), DefRef::Static(&DEF_c16));
  host.back.insert(Ptr::new_ref(&DEF_c16).loc(), r#"c16"#.to_owned());
  host.defs.insert(r#"c20"#.to_owned(), DefRef::Static(&DEF_c20));
  host.back.insert(Ptr::new_ref(&DEF_c20).loc(), r#"c20"#.to_owned());
  host.defs.insert(r#"c22"#.to_owned(), DefRef::Static(&DEF_c22));
  host.back.insert(Ptr::new_ref(&DEF_c22).loc(), r#"c22"#.to_owned());
  host.defs.insert(r#"c24"#.to_owned(), DefRef::Static(&DEF_c24));
  host.back.insert(Ptr::new_ref(&DEF_c24).loc(), r#"c24"#.to_owned());
  host.defs.insert(r#"c3"#.to_owned(), DefRef::Static(&DEF_c3));
  host.back.insert(Ptr::new_ref(&DEF_c3).loc(), r#"c3"#.to_owned());
  host.defs.insert(r#"c4"#.to_owned(), DefRef::Static(&DEF_c4));
  host.back.insert(Ptr::new_ref(&DEF_c4).loc(), r#"c4"#.to_owned());
  host.defs.insert(r#"c6"#.to_owned(), DefRef::Static(&DEF_c6));
  host.back.insert(Ptr::new_ref(&DEF_c6).loc(), r#"c6"#.to_owned());
  host.defs.insert(r#"c8"#.to_owned(), DefRef::Static(&DEF_c8));
  host.back.insert(Ptr::new_ref(&DEF_c8).loc(), r#"c8"#.to_owned());
  host.defs.insert(r#"dec"#.to_owned(), DefRef::Static(&DEF_dec));
  host.back.insert(Ptr::new_ref(&DEF_dec).loc(), r#"dec"#.to_owned());
  host.defs.insert(r#"decI"#.to_owned(), DefRef::Static(&DEF_decI));
  host.back.insert(Ptr::new_ref(&DEF_decI).loc(), r#"decI"#.to_owned());
  host.defs.insert(r#"decO"#.to_owned(), DefRef::Static(&DEF_decO));
  host.back.insert(Ptr::new_ref(&DEF_decO).loc(), r#"decO"#.to_owned());
  host.defs.insert(r#"low"#.to_owned(), DefRef::Static(&DEF_low));
  host.back.insert(Ptr::new_ref(&DEF_low).loc(), r#"low"#.to_owned());
  host.defs.insert(r#"lowI"#.to_owned(), DefRef::Static(&DEF_lowI));
  host.back.insert(Ptr::new_ref(&DEF_lowI).loc(), r#"lowI"#.to_owned());
  host.defs.insert(r#"lowO"#.to_owned(), DefRef::Static(&DEF_lowO));
  host.back.insert(Ptr::new_ref(&DEF_lowO).loc(), r#"lowO"#.to_owned());
  host.defs.insert(r#"main"#.to_owned(), DefRef::Static(&DEF_main));
  host.back.insert(Ptr::new_ref(&DEF_main).loc(), r#"main"#.to_owned());
  host.defs.insert(r#"run"#.to_owned(), DefRef::Static(&DEF_run));
  host.back.insert(Ptr::new_ref(&DEF_run).loc(), r#"run"#.to_owned());
  host.defs.insert(r#"runI"#.to_owned(), DefRef::Static(&DEF_runI));
  host.back.insert(Ptr::new_ref(&DEF_runI).loc(), r#"runI"#.to_owned());
  host.defs.insert(r#"runO"#.to_owned(), DefRef::Static(&DEF_runO));
  host.back.insert(Ptr::new_ref(&DEF_runO).loc(), r#"runO"#.to_owned());
  host
}

pub static DEF_decO: Def = Def { lab: 1, inner: DefType::Native(call_decO) };
pub static DEF_decI: Def = Def { lab: 1, inner: DefType::Native(call_decI) };
pub static DEF_c8: Def = Def { lab: 2, inner: DefType::Native(call_c8) };
pub static DEF_run: Def = Def { lab: 1, inner: DefType::Native(call_run) };
pub static DEF_c10: Def = Def { lab: 2, inner: DefType::Native(call_c10) };
pub static DEF_c20: Def = Def { lab: 2, inner: DefType::Native(call_c20) };
pub static DEF_c14: Def = Def { lab: 2, inner: DefType::Native(call_c14) };
pub static DEF_runO: Def = Def { lab: 1, inner: DefType::Native(call_runO) };
pub static DEF_I: Def = Def { lab: 1, inner: DefType::Native(call_I) };
pub static DEF_lowO: Def = Def { lab: 1, inner: DefType::Native(call_lowO) };
pub static DEF_Z: Def = Def { lab: 1, inner: DefType::Native(call_Z) };
pub static DEF_S: Def = Def { lab: 1, inner: DefType::Native(call_S) };
pub static DEF_runI: Def = Def { lab: 1, inner: DefType::Native(call_runI) };
pub static DEF_c16: Def = Def { lab: 2, inner: DefType::Native(call_c16) };
pub static DEF_brnZ: Def = Def { lab: 2, inner: DefType::Native(call_brnZ) };
pub static DEF_c24: Def = Def { lab: 2, inner: DefType::Native(call_c24) };
pub static DEF_brn: Def = Def { lab: 2, inner: DefType::Native(call_brn) };
pub static DEF_c4: Def = Def { lab: 2, inner: DefType::Native(call_c4) };
pub static DEF_low: Def = Def { lab: 1, inner: DefType::Native(call_low) };
pub static DEF_c6: Def = Def { lab: 2, inner: DefType::Native(call_c6) };
pub static DEF_main: Def = Def { lab: 2, inner: DefType::Native(call_main) };
pub static DEF_lowI: Def = Def { lab: 1, inner: DefType::Native(call_lowI) };
pub static DEF_dec: Def = Def { lab: 1, inner: DefType::Native(call_dec) };
pub static DEF_brnS: Def = Def { lab: 2, inner: DefType::Native(call_brnS) };
pub static DEF_O: Def = Def { lab: 1, inner: DefType::Native(call_O) };
pub static DEF_c12: Def = Def { lab: 2, inner: DefType::Native(call_c12) };
pub static DEF_c3: Def = Def { lab: 2, inner: DefType::Native(call_c3) };
pub static DEF_E: Def = Def { lab: 1, inner: DefType::Native(call_E) };
pub static DEF_c22: Def = Def { lab: 2, inner: DefType::Native(call_c22) };

pub fn call_E(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  net.half_safe_link(rtx, Ptr::ERA);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.half_safe_link(rtyx, Ptr::ERA);
  let (rtyyx, rtyyy) = net.quick_ctr(rtyy, 0);
  net.safe_link(rtyyx, rtyyy);
}

pub fn call_I(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.half_safe_link(rtyx, Ptr::ERA);
  let (rtyyx, rtyyy) = net.quick_ctr(rtyy, 0);
  let (rtyyxx, rtyyxy) = net.quick_ctr(rtyyx, 0);
  net.safe_link(rtx, rtyyxx);
  let (rtyyyx, rtyyyy) = net.quick_ctr(rtyyy, 0);
  net.half_safe_link(rtyyyx, Ptr::ERA);
  net.safe_link(rtyyxy, rtyyyy);
}

pub fn call_O(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  let (rtyxx, rtyxy) = net.quick_ctr(rtyx, 0);
  net.safe_link(rtx, rtyxx);
  let (rtyyx, rtyyy) = net.quick_ctr(rtyy, 0);
  net.half_safe_link(rtyyx, Ptr::ERA);
  let (rtyyyx, rtyyyy) = net.quick_ctr(rtyyy, 0);
  net.half_safe_link(rtyyyx, Ptr::ERA);
  net.safe_link(rtyxy, rtyyyy);
}

pub fn call_S(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  let (rtyxx, rtyxy) = net.quick_ctr(rtyx, 0);
  net.safe_link(rtx, rtyxx);
  let (rtyyx, rtyyy) = net.quick_ctr(rtyy, 0);
  net.half_safe_link(rtyyx, Ptr::ERA);
  net.safe_link(rtyxy, rtyyy);
}

pub fn call_Z(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  net.half_safe_link(rtx, Ptr::ERA);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtyx, rtyy);
}

pub fn call_brn(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 0);
  net.half_safe_link(rtxx, Ptr::new_ref(&DEF_brnS));
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.half_safe_link(rtxyx, Ptr::new_ref(&DEF_brnZ));
  net.safe_link(rtxyy, rty);
}

pub fn call_brnS(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_brn));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rtxx, rd_0_1x);
  net.safe_link(rtyx, rd_0_1y);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_brn));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rtxy, rd_1_1x);
  net.safe_link(rtyy, rd_1_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
}

pub fn call_brnZ(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  net.half_safe_link(rtx, Ptr::ERA);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_run));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rty, rd_0_1y);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_c20));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.half_safe_link(rd_1_1x, Ptr::new_ref(&DEF_I));
  let (rd_1_1yx, rd_1_1yy) = net.quick_ctr(rd_1_1y, 0);
  net.half_safe_link(rd_1_1yx, Ptr::new_ref(&DEF_E));
  net.safe_link(rd_0_1x, rd_1_1yy);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
}

pub fn call_c10(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 0);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c12(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxx, 0);
  let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxy, rtxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c14(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxx, 0);
  let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c16(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxx, 0);
  let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c20(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxx, 0);
  let (rtxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c22(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxx, 0);
  let (rtxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c24(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 1);
  let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxx, 1);
  let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxxx, 1);
  let (rtxxxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxxxy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxxxx, 0);
  let (rtxxxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
  let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
  let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
  let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.quick_ctr(rtxxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxxyy, rtxxxxxxxxyx);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxyy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxxxxxxxxxxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c3(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 0);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c4(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 0);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c6(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 0);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_c8(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 1);
  let (rtxxx, rtxxy) = net.quick_ctr(rtxx, 1);
  let (rtxxxx, rtxxxy) = net.quick_ctr(rtxxx, 1);
  let (rtxxxxx, rtxxxxy) = net.quick_ctr(rtxxxx, 1);
  let (rtxxxxxx, rtxxxxxy) = net.quick_ctr(rtxxxxx, 1);
  let (rtxxxxxxx, rtxxxxxxy) = net.quick_ctr(rtxxxxxx, 1);
  let (rtxxxxxxxx, rtxxxxxxxy) = net.quick_ctr(rtxxxxxxx, 1);
  let (rtxxxxxxxxx, rtxxxxxxxxy) = net.quick_ctr(rtxxxxxxxx, 0);
  let (rtxxxxxxxyx, rtxxxxxxxyy) = net.quick_ctr(rtxxxxxxxy, 0);
  net.safe_link(rtxxxxxxxxy, rtxxxxxxxyx);
  let (rtxxxxxxyx, rtxxxxxxyy) = net.quick_ctr(rtxxxxxxy, 0);
  net.safe_link(rtxxxxxxxyy, rtxxxxxxyx);
  let (rtxxxxxyx, rtxxxxxyy) = net.quick_ctr(rtxxxxxy, 0);
  net.safe_link(rtxxxxxxyy, rtxxxxxyx);
  let (rtxxxxyx, rtxxxxyy) = net.quick_ctr(rtxxxxy, 0);
  net.safe_link(rtxxxxxyy, rtxxxxyx);
  let (rtxxxyx, rtxxxyy) = net.quick_ctr(rtxxxy, 0);
  net.safe_link(rtxxxxyy, rtxxxyx);
  let (rtxxyx, rtxxyy) = net.quick_ctr(rtxxy, 0);
  net.safe_link(rtxxxyy, rtxxyx);
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.safe_link(rtxxyy, rtxyx);
  let (rtyx, rtyy) = net.quick_ctr(rty, 0);
  net.safe_link(rtxxxxxxxxx, rtyx);
  net.safe_link(rtxyy, rtyy);
}

pub fn call_dec(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 0);
  net.half_safe_link(rtxx, Ptr::new_ref(&DEF_decO));
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.half_safe_link(rtxyx, Ptr::new_ref(&DEF_decI));
  let (rtxyyx, rtxyyy) = net.quick_ctr(rtxyy, 0);
  net.half_safe_link(rtxyyx, Ptr::new_ref(&DEF_E));
  net.safe_link(rtxyyy, rty);
}

pub fn call_decI(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_low));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rtx, rd_0_1x);
  net.safe_link(rty, rd_0_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
}

pub fn call_decO(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_I));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rty, rd_0_1y);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_dec));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rtx, rd_1_1x);
  net.safe_link(rd_0_1x, rd_1_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
}

pub fn call_low(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 0);
  net.half_safe_link(rtxx, Ptr::new_ref(&DEF_lowO));
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.half_safe_link(rtxyx, Ptr::new_ref(&DEF_lowI));
  let (rtxyyx, rtxyyy) = net.quick_ctr(rtxyy, 0);
  net.half_safe_link(rtxyyx, Ptr::new_ref(&DEF_E));
  net.safe_link(rtxyyy, rty);
}

pub fn call_lowI(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_I));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rtx, rd_0_1x);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_O));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rd_0_1y, rd_1_1x);
  net.safe_link(rty, rd_1_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
}

pub fn call_lowO(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_O));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rtx, rd_0_1x);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_O));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rd_0_1y, rd_1_1x);
  net.safe_link(rty, rd_1_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
}

pub fn call_main(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_c4));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.half_safe_link(rd_0_1x, Ptr::new_ref(&DEF_S));
  let (rd_0_1yx, rd_0_1yy) = net.quick_ctr(rd_0_1y, 0);
  net.half_safe_link(rd_0_1yx, Ptr::new_ref(&DEF_Z));
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_brn));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rd_0_1yy, rd_1_1x);
  net.safe_link(rt, rd_1_1y);
  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
}

pub fn call_run(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let (rtxx, rtxy) = net.quick_ctr(rtx, 0);
  net.half_safe_link(rtxx, Ptr::new_ref(&DEF_runO));
  let (rtxyx, rtxyy) = net.quick_ctr(rtxy, 0);
  net.half_safe_link(rtxyx, Ptr::new_ref(&DEF_runI));
  let (rtxyyx, rtxyyy) = net.quick_ctr(rtxyy, 0);
  net.half_safe_link(rtxyyx, Ptr::new_ref(&DEF_E));
  net.safe_link(rtxyyy, rty);
}

pub fn call_runI(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_run));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rty, rd_0_1y);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_dec));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rd_0_1x, rd_1_1y);
  let mut rd_2_ = (Lcl::Todo(2), Lcl::Todo(2));
  let rd_2_0 = Trg::Lcl(&mut rd_2_.0);
  let rd_2_1 = Trg::Lcl(&mut rd_2_.1);
  net.half_safe_link(rd_2_0, Ptr::new_ref(&DEF_I));
  let (rd_2_1x, rd_2_1y) = net.quick_ctr(rd_2_1, 0);
  net.safe_link(rtx, rd_2_1x);
  net.safe_link(rd_1_1x, rd_2_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
  let (Lcl::Bound(rd_2_0), Lcl::Bound(rd_2_1)) = rd_2_ else { unreachable!() };
  net.safe_link(rd_2_0, rd_2_1);
}

pub fn call_runO(net: &mut Net, rt: Ptr) {
  let rt = Trg::Ptr(rt);
  let (rtx, rty) = net.quick_ctr(rt, 0);
  let mut rd_0_ = (Lcl::Todo(0), Lcl::Todo(0));
  let rd_0_0 = Trg::Lcl(&mut rd_0_.0);
  let rd_0_1 = Trg::Lcl(&mut rd_0_.1);
  net.half_safe_link(rd_0_0, Ptr::new_ref(&DEF_run));
  let (rd_0_1x, rd_0_1y) = net.quick_ctr(rd_0_1, 0);
  net.safe_link(rty, rd_0_1y);
  let mut rd_1_ = (Lcl::Todo(1), Lcl::Todo(1));
  let rd_1_0 = Trg::Lcl(&mut rd_1_.0);
  let rd_1_1 = Trg::Lcl(&mut rd_1_.1);
  net.half_safe_link(rd_1_0, Ptr::new_ref(&DEF_dec));
  let (rd_1_1x, rd_1_1y) = net.quick_ctr(rd_1_1, 0);
  net.safe_link(rd_0_1x, rd_1_1y);
  let mut rd_2_ = (Lcl::Todo(2), Lcl::Todo(2));
  let rd_2_0 = Trg::Lcl(&mut rd_2_.0);
  let rd_2_1 = Trg::Lcl(&mut rd_2_.1);
  net.half_safe_link(rd_2_0, Ptr::new_ref(&DEF_O));
  let (rd_2_1x, rd_2_1y) = net.quick_ctr(rd_2_1, 0);
  net.safe_link(rtx, rd_2_1x);
  net.safe_link(rd_1_1x, rd_2_1y);

  let (Lcl::Bound(rd_0_0), Lcl::Bound(rd_0_1)) = rd_0_ else { unreachable!() };
  net.safe_link(rd_0_0, rd_0_1);
  let (Lcl::Bound(rd_1_0), Lcl::Bound(rd_1_1)) = rd_1_ else { unreachable!() };
  net.safe_link(rd_1_0, rd_1_1);
  let (Lcl::Bound(rd_2_0), Lcl::Bound(rd_2_1)) = rd_2_ else { unreachable!() };
  net.safe_link(rd_2_0, rd_2_1);
}
