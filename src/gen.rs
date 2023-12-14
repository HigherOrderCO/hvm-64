// use crate::{
//   ast::{DefRef, Host},
//   jit::*,
//   run::*,
// };

// pub fn host() -> Host {
//   let mut host = Host::default();
//   host.defs.insert(r#"E"#.to_owned(), DefRef::Static(&DEF_E));
//   host.back.insert(Port::new_ref(&DEF_E).loc(), r#"E"#.to_owned());
//   host.defs.insert(r#"I"#.to_owned(), DefRef::Static(&DEF_I));
//   host.back.insert(Port::new_ref(&DEF_I).loc(), r#"I"#.to_owned());
//   host.defs.insert(r#"O"#.to_owned(), DefRef::Static(&DEF_O));
//   host.back.insert(Port::new_ref(&DEF_O).loc(), r#"O"#.to_owned());
//   host.defs.insert(r#"S"#.to_owned(), DefRef::Static(&DEF_S));
//   host.back.insert(Port::new_ref(&DEF_S).loc(), r#"S"#.to_owned());
//   host.defs.insert(r#"Z"#.to_owned(), DefRef::Static(&DEF_Z));
//   host.back.insert(Port::new_ref(&DEF_Z).loc(), r#"Z"#.to_owned());
//   host.defs.insert(r#"brn"#.to_owned(), DefRef::Static(&DEF_brn));
//   host.back.insert(Port::new_ref(&DEF_brn).loc(), r#"brn"#.to_owned());
//   host.defs.insert(r#"brnS"#.to_owned(), DefRef::Static(&DEF_brnS));
//   host.back.insert(Port::new_ref(&DEF_brnS).loc(), r#"brnS"#.to_owned());
//   host.defs.insert(r#"brnZ"#.to_owned(), DefRef::Static(&DEF_brnZ));
//   host.back.insert(Port::new_ref(&DEF_brnZ).loc(), r#"brnZ"#.to_owned());
//   host.defs.insert(r#"c10"#.to_owned(), DefRef::Static(&DEF_c10));
//   host.back.insert(Port::new_ref(&DEF_c10).loc(), r#"c10"#.to_owned());
//   host.defs.insert(r#"c12"#.to_owned(), DefRef::Static(&DEF_c12));
//   host.back.insert(Port::new_ref(&DEF_c12).loc(), r#"c12"#.to_owned());
//   host.defs.insert(r#"c14"#.to_owned(), DefRef::Static(&DEF_c14));
//   host.back.insert(Port::new_ref(&DEF_c14).loc(), r#"c14"#.to_owned());
//   host.defs.insert(r#"c16"#.to_owned(), DefRef::Static(&DEF_c16));
//   host.back.insert(Port::new_ref(&DEF_c16).loc(), r#"c16"#.to_owned());
//   host.defs.insert(r#"c20"#.to_owned(), DefRef::Static(&DEF_c20));
//   host.back.insert(Port::new_ref(&DEF_c20).loc(), r#"c20"#.to_owned());
//   host.defs.insert(r#"c22"#.to_owned(), DefRef::Static(&DEF_c22));
//   host.back.insert(Port::new_ref(&DEF_c22).loc(), r#"c22"#.to_owned());
//   host.defs.insert(r#"c24"#.to_owned(), DefRef::Static(&DEF_c24));
//   host.back.insert(Port::new_ref(&DEF_c24).loc(), r#"c24"#.to_owned());
//   host.defs.insert(r#"c3"#.to_owned(), DefRef::Static(&DEF_c3));
//   host.back.insert(Port::new_ref(&DEF_c3).loc(), r#"c3"#.to_owned());
//   host.defs.insert(r#"c4"#.to_owned(), DefRef::Static(&DEF_c4));
//   host.back.insert(Port::new_ref(&DEF_c4).loc(), r#"c4"#.to_owned());
//   host.defs.insert(r#"c6"#.to_owned(), DefRef::Static(&DEF_c6));
//   host.back.insert(Port::new_ref(&DEF_c6).loc(), r#"c6"#.to_owned());
//   host.defs.insert(r#"c8"#.to_owned(), DefRef::Static(&DEF_c8));
//   host.back.insert(Port::new_ref(&DEF_c8).loc(), r#"c8"#.to_owned());
//   host.defs.insert(r#"dec"#.to_owned(), DefRef::Static(&DEF_dec));
//   host.back.insert(Port::new_ref(&DEF_dec).loc(), r#"dec"#.to_owned());
//   host.defs.insert(r#"decI"#.to_owned(), DefRef::Static(&DEF_decI));
//   host.back.insert(Port::new_ref(&DEF_decI).loc(), r#"decI"#.to_owned());
//   host.defs.insert(r#"decO"#.to_owned(), DefRef::Static(&DEF_decO));
//   host.back.insert(Port::new_ref(&DEF_decO).loc(), r#"decO"#.to_owned());
//   host.defs.insert(r#"low"#.to_owned(), DefRef::Static(&DEF_low));
//   host.back.insert(Port::new_ref(&DEF_low).loc(), r#"low"#.to_owned());
//   host.defs.insert(r#"lowI"#.to_owned(), DefRef::Static(&DEF_lowI));
//   host.back.insert(Port::new_ref(&DEF_lowI).loc(), r#"lowI"#.to_owned());
//   host.defs.insert(r#"lowO"#.to_owned(), DefRef::Static(&DEF_lowO));
//   host.back.insert(Port::new_ref(&DEF_lowO).loc(), r#"lowO"#.to_owned());
//   host.defs.insert(r#"main"#.to_owned(), DefRef::Static(&DEF_main));
//   host.back.insert(Port::new_ref(&DEF_main).loc(), r#"main"#.to_owned());
//   host.defs.insert(r#"run"#.to_owned(), DefRef::Static(&DEF_run));
//   host.back.insert(Port::new_ref(&DEF_run).loc(), r#"run"#.to_owned());
//   host.defs.insert(r#"runI"#.to_owned(), DefRef::Static(&DEF_runI));
//   host.back.insert(Port::new_ref(&DEF_runI).loc(), r#"runI"#.to_owned());
//   host.defs.insert(r#"runO"#.to_owned(), DefRef::Static(&DEF_runO));
//   host.back.insert(Port::new_ref(&DEF_runO).loc(), r#"runO"#.to_owned());
//   host
// }

// pub static DEF_S: Def = Def { lab: 1, inner: DefType::Native(do_S) };
// pub static DEF_c20: Def = Def { lab: 2, inner: DefType::Native(do_c20) };
// pub static DEF_runO: Def = Def { lab: 1, inner: DefType::Native(do_runO) };
// pub static DEF_runI: Def = Def { lab: 1, inner: DefType::Native(do_runI) };
// pub static DEF_lowO: Def = Def { lab: 1, inner: DefType::Native(do_lowO) };
// pub static DEF_brnZ: Def = Def { lab: 2, inner: DefType::Native(do_brnZ) };
// pub static DEF_c8: Def = Def { lab: 2, inner: DefType::Native(do_c8) };
// pub static DEF_run: Def = Def { lab: 1, inner: DefType::Native(do_run) };
// pub static DEF_lowI: Def = Def { lab: 1, inner: DefType::Native(do_lowI) };
// pub static DEF_c10: Def = Def { lab: 2, inner: DefType::Native(do_c10) };
// pub static DEF_Z: Def = Def { lab: 1, inner: DefType::Native(do_Z) };
// pub static DEF_c24: Def = Def { lab: 2, inner: DefType::Native(do_c24) };
// pub static DEF_decI: Def = Def { lab: 1, inner: DefType::Native(do_decI) };
// pub static DEF_brn: Def = Def { lab: 2, inner: DefType::Native(do_brn) };
// pub static DEF_c3: Def = Def { lab: 2, inner: DefType::Native(do_c3) };
// pub static DEF_c22: Def = Def { lab: 2, inner: DefType::Native(do_c22) };
// pub static DEF_E: Def = Def { lab: 1, inner: DefType::Native(do_E) };
// pub static DEF_dec: Def = Def { lab: 1, inner: DefType::Native(do_dec) };
// pub static DEF_main: Def = Def { lab: 2, inner: DefType::Native(do_main) };
// pub static DEF_c6: Def = Def { lab: 2, inner: DefType::Native(do_c6) };
// pub static DEF_decO: Def = Def { lab: 1, inner: DefType::Native(do_decO) };
// pub static DEF_c12: Def = Def { lab: 2, inner: DefType::Native(do_c12) };
// pub static DEF_I: Def = Def { lab: 1, inner: DefType::Native(do_I) };
// pub static DEF_low: Def = Def { lab: 1, inner: DefType::Native(do_low) };
// pub static DEF_brnS: Def = Def { lab: 2, inner: DefType::Native(do_brnS) };
// pub static DEF_c14: Def = Def { lab: 2, inner: DefType::Native(do_c14) };
// pub static DEF_c16: Def = Def { lab: 2, inner: DefType::Native(do_c16) };
// pub static DEF_O: Def = Def { lab: 1, inner: DefType::Native(do_O) };
// pub static DEF_c4: Def = Def { lab: 2, inner: DefType::Native(do_c4) };

// pub fn do_E(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   net.link_trg_ptr(rtx, Port::ERA);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg_ptr(rtyx, Port::ERA);
//   let (rtyyx, rtyyy) = net.do_ctr(rtyy, 0);
//   net.link_trg(rtyyx, rtyyy);
// }

// pub fn do_I(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg_ptr(rtyx, Port::ERA);
//   let (rtyyx, rtyyy) = net.do_ctr(rtyy, 0);
//   let (rtyyxx, rtyyxy) = net.do_ctr(rtyyx, 0);
//   net.link_trg(rtx, rtyyxx);
//   let (rtyyyx, rtyyyy) = net.do_ctr(rtyyy, 0);
//   net.link_trg_ptr(rtyyyx, Port::ERA);
//   net.link_trg(rtyyxy, rtyyyy);
// }

// pub fn do_O(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   let (rtyxx, rtyxy) = net.do_ctr(rtyx, 0);
//   net.link_trg(rtx, rtyxx);
//   let (rtyyx, rtyyy) = net.do_ctr(rtyy, 0);
//   net.link_trg_ptr(rtyyx, Port::ERA);
//   let (rtyyyx, rtyyyy) = net.do_ctr(rtyyy, 0);
//   net.link_trg_ptr(rtyyyx, Port::ERA);
//   net.link_trg(rtyxy, rtyyyy);
// }

// pub fn do_S(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   let (rtyxx, rtyxy) = net.do_ctr(rtyx, 0);
//   net.link_trg(rtx, rtyxx);
//   let (rtyyx, rtyyy) = net.do_ctr(rtyy, 0);
//   net.link_trg_ptr(rtyyx, Port::ERA);
//   net.link_trg(rtyxy, rtyyy);
// }

// pub fn do_Z(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   net.link_trg_ptr(rtx, Port::ERA);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtyx, rtyy);
// }

// pub fn do_brn(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 0);
//   net.link_trg_ptr(rtxx, Port::new_ref(&DEF_brnS));
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg_ptr(rtxyx, Port::new_ref(&DEF_brnZ));
//   net.link_trg(rtxyy, rty);
// }

// pub fn do_brnS(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_brn));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rtxx, rd0x);
//   net.link_trg(rtyx, rd0y);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_brn));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rtxy, rd1x);
//   net.link_trg(rtyy, rd1y);
// }

// pub fn do_brnZ(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   net.link_trg_ptr(rtx, Port::ERA);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_run));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rty, rd0y);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_c20));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg_ptr(rd1x, Port::new_ref(&DEF_I));
//   let (rd1yx, rd1yy) = net.do_ctr(rd1y, 0);
//   net.link_trg_ptr(rd1yx, Port::new_ref(&DEF_E));
//   net.link_trg(rd0x, rd1yy);
// }

// pub fn do_c10(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c12(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxy, rtxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c14(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c16(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c20(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c22(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c24(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 1);
//   let (rtxxxxxxxxxx, rtxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxx, rtxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxx, rtxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxx, rtxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxxx, 1);
//   let (rtxxxxxxxxxxxxxxxxxxxxxxxxx, rtxxxxxxxxxxxxxxxxxxxxxxxxy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxxxx, 0);
//   let (rtxxxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxxxxy, rtxxxxxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxxyx, rtxxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxxyx, rtxxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxxyy, rtxxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxxyx, rtxxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxxyy, rtxxxxxxxxxxxyx);
//   let (rtxxxxxxxxxxyx, rtxxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxxyy, rtxxxxxxxxxxyx);
//   let (rtxxxxxxxxxyx, rtxxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxxyy, rtxxxxxxxxxyx);
//   let (rtxxxxxxxxyx, rtxxxxxxxxyy) = net.do_ctr(rtxxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxxyy, rtxxxxxxxxyx);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxyy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxxxxxxxxxxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c3(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 0);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c4(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 0);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c6(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 0);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_c8(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 1);
//   let (rtxxx, rtxxy) = net.do_ctr(rtxx, 1);
//   let (rtxxxx, rtxxxy) = net.do_ctr(rtxxx, 1);
//   let (rtxxxxx, rtxxxxy) = net.do_ctr(rtxxxx, 1);
//   let (rtxxxxxx, rtxxxxxy) = net.do_ctr(rtxxxxx, 1);
//   let (rtxxxxxxx, rtxxxxxxy) = net.do_ctr(rtxxxxxx, 1);
//   let (rtxxxxxxxx, rtxxxxxxxy) = net.do_ctr(rtxxxxxxx, 1);
//   let (rtxxxxxxxxx, rtxxxxxxxxy) = net.do_ctr(rtxxxxxxxx, 0);
//   let (rtxxxxxxxyx, rtxxxxxxxyy) = net.do_ctr(rtxxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxxy, rtxxxxxxxyx);
//   let (rtxxxxxxyx, rtxxxxxxyy) = net.do_ctr(rtxxxxxxy, 0);
//   net.link_trg(rtxxxxxxxyy, rtxxxxxxyx);
//   let (rtxxxxxyx, rtxxxxxyy) = net.do_ctr(rtxxxxxy, 0);
//   net.link_trg(rtxxxxxxyy, rtxxxxxyx);
//   let (rtxxxxyx, rtxxxxyy) = net.do_ctr(rtxxxxy, 0);
//   net.link_trg(rtxxxxxyy, rtxxxxyx);
//   let (rtxxxyx, rtxxxyy) = net.do_ctr(rtxxxy, 0);
//   net.link_trg(rtxxxxyy, rtxxxyx);
//   let (rtxxyx, rtxxyy) = net.do_ctr(rtxxy, 0);
//   net.link_trg(rtxxxyy, rtxxyx);
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg(rtxxyy, rtxyx);
//   let (rtyx, rtyy) = net.do_ctr(rty, 0);
//   net.link_trg(rtxxxxxxxxx, rtyx);
//   net.link_trg(rtxyy, rtyy);
// }

// pub fn do_dec(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 0);
//   net.link_trg_ptr(rtxx, Port::new_ref(&DEF_decO));
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg_ptr(rtxyx, Port::new_ref(&DEF_decI));
//   let (rtxyyx, rtxyyy) = net.do_ctr(rtxyy, 0);
//   net.link_trg_ptr(rtxyyx, Port::new_ref(&DEF_E));
//   net.link_trg(rtxyyy, rty);
// }

// pub fn do_decI(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_low));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rtx, rd0x);
//   net.link_trg(rty, rd0y);
// }

// pub fn do_decO(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_I));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rty, rd0y);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_dec));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rtx, rd1x);
//   net.link_trg(rd0x, rd1y);
// }

// pub fn do_low(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 0);
//   net.link_trg_ptr(rtxx, Port::new_ref(&DEF_lowO));
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg_ptr(rtxyx, Port::new_ref(&DEF_lowI));
//   let (rtxyyx, rtxyyy) = net.do_ctr(rtxyy, 0);
//   net.link_trg_ptr(rtxyyx, Port::new_ref(&DEF_E));
//   net.link_trg(rtxyyy, rty);
// }

// pub fn do_lowI(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_I));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rtx, rd0x);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_O));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rd0y, rd1x);
//   net.link_trg(rty, rd1y);
// }

// pub fn do_lowO(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_O));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rtx, rd0x);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_O));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rd0y, rd1x);
//   net.link_trg(rty, rd1y);
// }

// pub fn do_main(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_c6));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg_ptr(rd0x, Port::new_ref(&DEF_S));
//   let (rd0yx, rd0yy) = net.do_ctr(rd0y, 0);
//   net.link_trg_ptr(rd0yx, Port::new_ref(&DEF_Z));
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_brn));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rd0yy, rd1x);
//   net.link_trg(rt, rd1y);
// }

// pub fn do_run(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let (rtxx, rtxy) = net.do_ctr(rtx, 0);
//   net.link_trg_ptr(rtxx, Port::new_ref(&DEF_runO));
//   let (rtxyx, rtxyy) = net.do_ctr(rtxy, 0);
//   net.link_trg_ptr(rtxyx, Port::new_ref(&DEF_runI));
//   let (rtxyyx, rtxyyy) = net.do_ctr(rtxyy, 0);
//   net.link_trg_ptr(rtxyyx, Port::new_ref(&DEF_E));
//   net.link_trg(rtxyyy, rty);
// }

// pub fn do_runI(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_run));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rty, rd0y);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_dec));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rd0x, rd1y);
//   let rd2 = Trg::Ptr(Port::new_ref(&DEF_I));
//   let (rd2x, rd2y) = net.do_ctr(rd2, 0);
//   net.link_trg(rtx, rd2x);
//   net.link_trg(rd1x, rd2y);
// }

// pub fn do_runO(net: &mut Net, rt: Port) {
//   let rt = Trg::Ptr(rt);
//   let (rtx, rty) = net.do_ctr(rt, 0);
//   let rd0 = Trg::Ptr(Port::new_ref(&DEF_run));
//   let (rd0x, rd0y) = net.do_ctr(rd0, 0);
//   net.link_trg(rty, rd0y);
//   let rd1 = Trg::Ptr(Port::new_ref(&DEF_dec));
//   let (rd1x, rd1y) = net.do_ctr(rd1, 0);
//   net.link_trg(rd0x, rd1y);
//   let rd2 = Trg::Ptr(Port::new_ref(&DEF_O));
//   let (rd2x, rd2y) = net.do_ctr(rd2, 0);
//   net.link_trg(rtx, rd2x);
//   net.link_trg(rd1x, rd2y);
// }
