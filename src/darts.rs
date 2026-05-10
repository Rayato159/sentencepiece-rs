use crate::{Error, Result};

#[derive(Clone, Debug)]
pub(crate) struct DoubleArray {
    units: Vec<Unit>,
}

#[derive(Clone, Copy, Debug)]
struct Unit(u32);

impl Unit {
    fn has_leaf(self) -> bool {
        ((self.0 >> 8) & 1) == 1
    }

    fn value(self) -> i32 {
        (self.0 & ((1 << 31) - 1)) as i32
    }

    fn label(self) -> u32 {
        self.0 & ((1 << 31) | 0xff)
    }

    fn offset(self) -> usize {
        ((self.0 >> 10) << ((self.0 & (1 << 9)) >> 6)) as usize
    }
}

impl DoubleArray {
    pub(crate) fn from_le_blob(blob: &[u8]) -> Result<Self> {
        if !blob.len().is_multiple_of(4) {
            return Err(Error::model_parse(
                "double-array trie blob length is not divisible by 4",
            ));
        }

        let units = blob
            .chunks_exact(4)
            .map(|chunk| Unit(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
            .collect::<Vec<_>>();

        let trie = Self { units };
        if !trie.validate() {
            return Err(Error::model_parse(
                "double-array trie contains out-of-bounds references",
            ));
        }
        Ok(trie)
    }

    pub(crate) fn common_prefix_search(&self, key: &[u8]) -> Vec<(i32, usize)> {
        let mut results = Vec::new();
        if self.units.is_empty() {
            return results;
        }

        let mut unit = self.units[0];
        let mut node_pos = unit.offset();
        for (index, byte) in key.iter().copied().enumerate() {
            node_pos ^= byte as usize;
            let Some(next_unit) = self.units.get(node_pos).copied() else {
                return results;
            };
            unit = next_unit;
            if unit.label() != byte as u32 {
                return results;
            }

            node_pos ^= unit.offset();
            if unit.has_leaf()
                && let Some(value_unit) = self.units.get(node_pos).copied()
            {
                results.push((value_unit.value(), index + 1));
            }
        }

        results
    }

    fn validate(&self) -> bool {
        let size = self.units.len();
        for (index, unit) in self.units.iter().copied().enumerate() {
            if unit.label() > 0xff {
                continue;
            }

            let offset = unit.offset();
            if offset == 0 {
                continue;
            }

            let base = index ^ offset;
            if (base | 0xff) >= size {
                return false;
            }
        }
        true
    }
}
