use crate::prelude::*;
use std::cell::RefCell;
use std::iter::FromIterator;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

thread_local! {
    static POOL: RefCell<Vec<Vec<Literal>>> = RefCell::new(vec![])
}

struct ClauseLiterals {
    literals: ManuallyDrop<Vec<Literal>>,
}

impl ClauseLiterals {
    fn new(literals: Vec<Literal>) -> Self {
        let literals = ManuallyDrop::new(literals);
        Self { literals }
    }

    fn recycle() -> Self {
        POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            let mut literals = pool.pop().unwrap_or_default();
            literals.clear();
            Self::new(literals)
        })
    }
}

impl Deref for ClauseLiterals {
    type Target = Vec<Literal>;
    fn deref(&self) -> &Self::Target {
        &self.literals
    }
}

impl DerefMut for ClauseLiterals {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.literals
    }
}

impl Clone for ClauseLiterals {
    fn clone(&self) -> Self {
        let mut recycled = Self::recycle();
        recycled.extend_from_slice(&self.literals);
        recycled
    }
}

impl Drop for ClauseLiterals {
    fn drop(&mut self) {
        let literals = unsafe { ManuallyDrop::take(&mut self.literals) };
        POOL.with(|pool| {
            let mut pool = pool.borrow_mut();
            pool.push(literals);
        });
    }
}

#[derive(Clone)]
pub struct Clause {
    literals: ClauseLiterals,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        let literals = ClauseLiterals::new(literals);
        Self { literals }
    }

    pub fn offset(&mut self, offset: Offset<Term>) {
        for literal in &mut *self.literals {
            literal.offset(offset);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Literal> {
        self.literals.iter()
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn last_literal(&self) -> &Literal {
        self.literals.last().unwrap()
    }

    pub fn pop_literal(&mut self) -> Literal {
        self.literals.pop().unwrap()
    }

    pub fn remove_literal(&mut self, literal_id: Id<Literal>) -> Literal {
        self.literals.remove(literal_id.index())
    }
}

impl FromIterator<Literal> for Clause {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Literal>,
    {
        let mut literals = ClauseLiterals::recycle();
        literals.extend(iter);
        Self { literals }
    }
}
