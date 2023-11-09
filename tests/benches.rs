use hvmc::ast::show_net;
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

pub fn load_bench(file: &str) -> String {
  let path = format!("{}/benches/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  std::fs::read_to_string(path).unwrap()
}

// Loads file and generate Book from hvm-core syntax
fn load_core(file: &str) -> hvmc::ast::Book {
  let code = load_bench(file);
  parse_core(&code)
}

// Loads file and generate DefinitionBook from hvm-lang syntax
pub fn load_lang(file: &str) -> hvm_lang::term::DefinitionBook {
  let code = load_bench(file);
  parse_lang(&code)
}

#[test]
#[cfg(not(feature = "cuda"))] // FIXME: hangs indefinitely
fn dec_bits() {
  let book = load_core("binary-counter/dec_bits.hvmc");
  let (rnet, net) = normal(book, 1 << 16);

  assert_snapshot!(show_net(&net), @"(* (* (b b)))");
  assert_debug_snapshot!(rnet.rewrites(), @"180113");
}

#[test]
fn dec_bits_tree() {
  let book = load_core("binary-counter/dec_bits_tree.hvmc");
  let (rnet, net) = normal(book, 1 << 13);

  // FIXME: The refs to the rule `@E` are not expanded
  if cfg!(feature = "cuda") {
    assert_snapshot!(show_net(&net), @"((((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E)))) ((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E))))) (((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E)))) ((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E))))))");
    assert_debug_snapshot!(rnet.rewrites(), @"2878529");
  } else {
    assert_snapshot!(show_net(&net), @"(((((((* (* (b b))) (* (* (c c)))) ((* (* (d d))) (* (* (e e))))) (((* (* (f f))) (* (* (g g)))) ((* (* (h h))) (* (* (i i)))))) ((((* (* (j j))) (* (* (k k)))) ((* (* (l l))) (* (* (m m))))) (((* (* (n n))) (* (* (o o)))) ((* (* (p p))) (* (* (q q))))))) (((((* (* (r r))) (* (* (s s)))) ((* (* (t t))) (* (* (u u))))) (((* (* (v v))) (* (* (w w)))) ((* (* (x x))) (* (* (y y)))))) ((((* (* (z z))) (* (* (ba ba)))) ((* (* (bb bb))) (* (* (bc bc))))) (((* (* (bd bd))) (* (* (be be)))) ((* (* (bf bf))) (* (* (bg bg)))))))) ((((((* (* (bh bh))) (* (* (bi bi)))) ((* (* (bj bj))) (* (* (bk bk))))) (((* (* (bl bl))) (* (* (bm bm)))) ((* (* (bn bn))) (* (* (bo bo)))))) ((((* (* (bp bp))) (* (* (bq bq)))) ((* (* (br br))) (* (* (bs bs))))) (((* (* (bt bt))) (* (* (bu bu)))) ((* (* (bv bv))) (* (* (bw bw))))))) (((((* (* (bx bx))) (* (* (by by)))) ((* (* (bz bz))) (* (* (ca ca))))) (((* (* (cb cb))) (* (* (cc cc)))) ((* (* (cd cd))) (* (* (ce ce)))))) ((((* (* (cf cf))) (* (* (cg cg)))) ((* (* (ch ch))) (* (* (ci ci))))) (((* (* (cj cj))) (* (* (ck ck)))) ((* (* (cl cl))) (* (* (cm cm)))))))))");
    assert_debug_snapshot!(rnet.rewrites(), @"2878593");
  }
}

#[test]
#[cfg(not(feature = "cuda"))] // FIXME: gpu runtime panics with `CUDA_ERROR_ILLEGAL_ADDRESS`
fn test_church_exp() {
  let book = load_core("church/church_exp.hvmc");
  let (rnet, net) = normal(book, 1 << 12);

  assert_snapshot!(show_net(&net), @"({2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 ({2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 b c} d} e} f} g} h} i} j} k} l} m} n} o} p} q} r} s} t} u} v} w} x} y} z} ba} bb} bc} bd} be} bf} bg} bh} bi} bj} bk} bl} bm} bn} bo} bp} bq} br} bs} bt} bu} bv} bw} bx} by} bz} ca} cb} cc} cd} ce} cf} cg} ch} ci} cj} ck} cl} cm} cn} co} cp} cq} cr} cs} ct} cu} cv} cw} cx} cy} cz} da} db} dc} dd} de} df} dg} dh} di} dj} dk} dl} dm} dn} do} dp} dq} dr} ds} dt} du} dv} dw} dx} dy} dz} ea} eb} ec} ed} ee} ef} eg} eh} ei} ej} ek} el} em} en} eo} ep} eq} er} es} et} eu} ev} ew} ex} ey} ez} fa} fb} fc} fd} fe} ff} fg} fh} fi} fj} fk} fl} fm} fn} fo} fp} fq} fr} fs} ft} fu} fv} fw} fx} fy} fz} ga} gb} gc} gd} ge} gf} gg} gh} gi} gj} gk} gl} gm} gn} go} gp} gq} gr} gs} gt} gu} gv} gw} gx} gy} gz} ha} hb} hc} hd} he} hf} hg} hh} hi} hj} hk} hl} hm} hn} ho} hp} hq} hr} hs} ht} hu} hv} hw} hx} hy} hz} ia} ib} ic} id} ie} if} ig} ih} ii} ij} ik} il} im} in} io} ip} iq} ir} is} it} iu} iv} iw} ix} iy} iz} ja} jb} jc} jd} je} jf} jg} jh} ji} jj} jk} jl} jm} jn} jo} jp} jq} jr} js} jt} ju} jv} jw} jx} jy} jz} ka} kb} kc} kd} ke} kf} kg} kh} ki} kj} kk} kl} km} kn} ko} kp} kq} kr} ks} kt} ku} kv} kw} kx} ky} kz} la} lb} lc} ld} le} lf} lg} lh} li} lj} lk} ll} lm} ln} lo} lp} lq} lr} ls} lt} lu} lv} lw} lx} ly} lz} ma} mb} mc} md} me} mf} mg} mh} mi} mj} mk} ml} mm} mn} mo} mp} mq} mr} ms} mt} mu} mv} mw} mx} my} mz} na} nb} nc} nd} ne} nf} ng} nh} ni} nj} nk} nl} nm} nn} no} np} nq} nr} ns} nt} nu} nv} nw} nx} ny} nz) (nz oa)} (oa ob)} (ob oc)} (oc od)} (od oe)} (oe of)} (of og)} (og oh)} (oh oi)} (oi oj)} (oj ok)} (ok ol)} (ol om)} (om on)} (on oo)} (oo op)} (op oq)} (oq or)} (or {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 ng nh} ni} nj} nk} nl} nm} nn} no} np} nq} nr} ns} nt} nu} nv} nw} nx} ny} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 mn mo} mp} mq} mr} ms} mt} mu} mv} mw} mx} my} mz} na} nb} nc} nd} ne} nf} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 lu lv} lw} lx} ly} lz} ma} mb} mc} md} me} mf} mg} mh} mi} mj} mk} ml} mm} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 lb lc} ld} le} lf} lg} lh} li} lj} lk} ll} lm} ln} lo} lp} lq} lr} ls} lt} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 ki kj} kk} kl} km} kn} ko} kp} kq} kr} ks} kt} ku} kv} kw} kx} ky} kz} la} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 jp jq} jr} js} jt} ju} jv} jw} jx} jy} jz} ka} kb} kc} kd} ke} kf} kg} kh} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 iw ix} iy} iz} ja} jb} jc} jd} je} jf} jg} jh} ji} jj} jk} jl} jm} jn} jo} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 id ie} if} ig} ih} ii} ij} ik} il} im} in} io} ip} iq} ir} is} it} iu} iv} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 hk hl} hm} hn} ho} hp} hq} hr} hs} ht} hu} hv} hw} hx} hy} hz} ia} ib} ic} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 gr gs} gt} gu} gv} gw} gx} gy} gz} ha} hb} hc} hd} he} hf} hg} hh} hi} hj} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 fy fz} ga} gb} gc} gd} ge} gf} gg} gh} gi} gj} gk} gl} gm} gn} go} gp} gq} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 ff fg} fh} fi} fj} fk} fl} fm} fn} fo} fp} fq} fr} fs} ft} fu} fv} fw} fx} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 em en} eo} ep} eq} er} es} et} eu} ev} ew} ex} ey} ez} fa} fb} fc} fd} fe} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 dt du} dv} dw} dx} dy} dz} ea} eb} ec} ed} ee} ef} eg} eh} ei} ej} ek} el} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 da db} dc} dd} de} df} dg} dh} di} dj} dk} dl} dm} dn} do} dp} dq} dr} ds} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 ch ci} cj} ck} cl} cm} cn} co} cp} cq} cr} cs} ct} cu} cv} cw} cx} cy} cz} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 bo bp} bq} br} bs} bt} bu} bv} bw} bx} by} bz} ca} cb} cc} cd} ce} cf} cg} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 v w} x} y} z} ba} bb} bc} bd} be} bf} bg} bh} bi} bj} bk} bl} bm} bn} {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 {2 c d} e} f} g} h} i} j} k} l} m} n} o} p} q} r} s} t} u} os}}}}}}}}}}}}}}}}}}})} (b os))");
  assert_debug_snapshot!(rnet.rewrites(), @"1943");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church/church_mul.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert_debug_snapshot!(valid_readback, @"false"); // invalid because of dup labels 
  assert_snapshot!(show_net(&net), @"({2 ({2 b {3 c {4 d {5 e f}}}} g) {3 (g h) {4 (h i) {5 (i j) k}}}} (b l))");
  assert_snapshot!(readback, @"λa λb ((* {λc d {λd e {λe f {λf (a (a (a (a {b {c {c {c c}}}})))) a}}}}) c)");
  assert_debug_snapshot!(rnet.rewrites(), @"17");
}

#[test]
#[cfg(not(feature = "cuda"))] // FIXME: hangs indefinitely
fn alloc_big_tree() {
  let book = load_core("tree/alloc_big_tree.hvmc");
  let (rnet, net) = normal(book, 1 << 16);

  assert_snapshot!(show_net(&net)); // file snapshot
  assert_debug_snapshot!(rnet.rewrites(), @"24628");
}

#[test]
fn test_neg_fusion() {
  let mut book = load_lang("fusion/neg_fusion.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"(b (* b))");
  assert_snapshot!(readback, @"λa λ* a");

  // TODO: investigate why this difference exists
  if cfg!(feature = "cuda") {
    assert_debug_snapshot!(rnet.rewrites(), @"160");
  } else {
    assert_debug_snapshot!(rnet.rewrites(), @"153");
  }
}
