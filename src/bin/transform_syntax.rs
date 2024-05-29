use core::fmt::Write;
use dyntest::{dyntest, DynTester};
use hvm64_ast::parser::Hvm64Parser;
use std::fs;
use TSPL::Parser;

dyntest!(transform);

fn transform(t: &mut DynTester) {
  for (name, path) in t.glob("{examples,tests/programs}/**/*.hvm") {
    t.test(name, move || {
      let file = fs::read_to_string(&path).unwrap();
      let mut transformer = Transformer { parser: Hvm64Parser::new(&file), output: String::new() };
      transformer.transform_book();
      fs::write(&path, transformer.output).unwrap();
    });
  }
}

struct Transformer<'a> {
  parser: Hvm64Parser<'a>,
  output: String,
}

impl Transformer<'_> {
  fn preserve(&mut self, f: impl FnOnce(&mut Hvm64Parser)) {
    let start = *self.parser.index();
    f(&mut self.parser);
    let end = *self.parser.index();
    self.output.write_str(&self.parser.input()[start .. end]).unwrap();
  }

  fn preserve_trivia(&mut self) {
    self.preserve(|p| p.skip_trivia())
  }

  fn consume(&mut self, str: &str) {
    self.preserve(|f| f.consume(str).unwrap());
  }

  fn try_consume(&mut self, str: &str) -> bool {
    self.preserve_trivia();
    if self.parser.peek_many(str.len()) == Some(str) {
      self.consume(str);
      true
    } else {
      false
    }
  }

  fn transform_book(&mut self) {
    self.preserve_trivia();
    while self.try_consume("@") {
      self.preserve(|p| {
        p.parse_name().unwrap();
      });
      self.preserve_trivia();
      self.consume("=");
      self.preserve_trivia();
      self.transform_net();
      self.preserve_trivia();
    }
  }

  fn transform_net(&mut self) {
    self.preserve_trivia();
    self.transform_tree();
    self.preserve_trivia();
    while self.try_consume("&") {
      self.preserve_trivia();
      self.transform_tree();
      self.preserve_trivia();
      self.consume("~");
      self.preserve_trivia();
      self.transform_tree();
      self.preserve_trivia();
    }
  }

  fn transform_tree(&mut self) {
    let tree = self.parser.parse_tree().unwrap();
    write!(self.output, "{}", tree).unwrap();
  }
}
