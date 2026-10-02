use crate::protocol::{Violation, ViolationKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalcOutcome {
    Result(String),
    Violation(Violation),
}

#[derive(Debug, Clone, Default)]
pub struct CalculatorTurn {
    requests: u32,
}

impl CalculatorTurn {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn evaluate(&mut self, expression: &str) -> CalcOutcome {
        self.requests += 1;
        if self.requests > 3 {
            return CalcOutcome::Violation(Violation {
                kind: ViolationKind::Format,
                attempt: 1,
                message: "calculator may be used at most three times per turn".to_string(),
                raw_output: expression.to_string(),
            });
        }
        match eval_integer_expression(expression) {
            Ok(value) => CalcOutcome::Result(value.to_string()),
            Err(reason) => CalcOutcome::Result(format!("error: {reason}")),
        }
    }
}

pub fn eval_integer_expression(expression: &str) -> Result<i128, String> {
    if expression.chars().count() > 200 {
        return Err("expression exceeds 200 characters".to_string());
    }
    let mut parser = Parser::new(expression);
    let value = parser.parse_expression()?;
    parser.skip_spaces();
    if parser.position != parser.bytes.len() {
        return Err("unexpected trailing input".to_string());
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            position: 0,
        }
    }

    fn skip_spaces(&mut self) {
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_whitespace)
        {
            self.position += 1;
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        self.skip_spaces();
        if self.bytes.get(self.position) == Some(&byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn parse_expression(&mut self) -> Result<i128, String> {
        let mut value = self.parse_term()?;
        loop {
            if self.consume(b'+') {
                value = value
                    .checked_add(self.parse_term()?)
                    .ok_or_else(|| "integer overflow".to_string())?;
            } else if self.consume(b'-') {
                value = value
                    .checked_sub(self.parse_term()?)
                    .ok_or_else(|| "integer overflow".to_string())?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_term(&mut self) -> Result<i128, String> {
        let mut value = self.parse_unary()?;
        loop {
            if self.consume(b'*') {
                value = value
                    .checked_mul(self.parse_unary()?)
                    .ok_or_else(|| "integer overflow".to_string())?;
            } else if self.consume(b'/') {
                let divisor = self.parse_unary()?;
                if divisor == 0 {
                    return Err("division by zero".to_string());
                }
                value = value
                    .checked_div_euclid(divisor)
                    .ok_or_else(|| "integer overflow".to_string())?;
            } else if self.consume(b'%') {
                let divisor = self.parse_unary()?;
                if divisor == 0 {
                    return Err("remainder by zero".to_string());
                }
                value = value
                    .checked_rem_euclid(divisor)
                    .ok_or_else(|| "integer overflow".to_string())?;
            } else {
                return Ok(value);
            }
        }
    }

    fn parse_unary(&mut self) -> Result<i128, String> {
        if self.consume(b'-') {
            if let Some(magnitude) = self.parse_unsigned_literal()? {
                let minimum_magnitude = (i128::MAX as u128) + 1;
                if magnitude == minimum_magnitude {
                    return Ok(i128::MIN);
                }
                let value =
                    i128::try_from(magnitude).map_err(|_| "integer overflow".to_string())?;
                return value
                    .checked_neg()
                    .ok_or_else(|| "integer overflow".to_string());
            }
            return self
                .parse_unary()?
                .checked_neg()
                .ok_or_else(|| "integer overflow".to_string());
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<i128, String> {
        if self.consume(b'(') {
            let value = self.parse_expression()?;
            if !self.consume(b')') {
                return Err("missing closing parenthesis".to_string());
            }
            return Ok(value);
        }
        self.skip_spaces();
        let Some(value) = self.parse_unsigned_literal()? else {
            return Err("expected an integer or parenthesis".to_string());
        };
        i128::try_from(value).map_err(|_| "integer overflow".to_string())
    }

    fn parse_unsigned_literal(&mut self) -> Result<Option<u128>, String> {
        self.skip_spaces();
        let start = self.position;
        while self
            .bytes
            .get(self.position)
            .is_some_and(u8::is_ascii_digit)
        {
            self.position += 1;
        }
        if start == self.position {
            return Ok(None);
        }
        let value = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| "invalid integer".to_string())?
            .parse::<u128>()
            .map_err(|_| "integer overflow".to_string())?;
        Ok(Some(value))
    }
}
