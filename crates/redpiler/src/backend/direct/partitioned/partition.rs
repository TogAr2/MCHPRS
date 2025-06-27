use std::ops::{Index, IndexMut};
use std::sync::{Arc, Mutex};
use itertools::Itertools;
use crate::backend::direct::node::{ForwardLink, ForwardLinkData, NodeId};
use crate::backend::direct::partitioned::Partition;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeLocation {
    partition: u16,
    index: NodeId,
}

impl NodeLocation {
    pub fn partition(&self) -> usize {
        self.partition as usize
    }

    pub fn index(&self) -> NodeId {
        self.index
    }

    /// Safety: partition must exist and index must be within its bounds
    pub unsafe fn from(partition: usize, index: usize) -> Self {
        Self {
            partition: partition as u16,
            index: NodeId::from_index(index)
        }
    }
}

impl PartialOrd for NodeLocation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeLocation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let ordering = self.partition.cmp(&other.partition);
        if ordering.is_ne() {
            ordering
        } else {
            self.index.cmp(&other.index)
        }
    }
}

#[derive(Copy, Clone)]
pub struct PartitionForwardLink {
    data: ForwardLinkData,
}

impl PartitionForwardLink {
    pub fn new(location: NodeLocation, side: bool, ss: u8) -> Self {
        assert!(location.partition() < (1 << 7));
        assert!(location.index().index() < (1 << 20));
        assert!(ss < 15);
        let partition = (location.partition() as u32) << 25;
        let index = (location.index().index() as u32) << 5;
        let side = if side { 1 << 4 } else { 0 };
        unsafe {
            // Safety: data is created from a NodeLocation
            Self {
                data: ForwardLinkData::new(partition | index | side | ss as u32)
            }
        }
    }

    pub fn data(self) -> ForwardLinkData {
        self.data
    }

    pub fn inner(self) -> u32 {
        self.data.inner()
    }
}

impl From<ForwardLinkData> for PartitionForwardLink {
    fn from(data: ForwardLinkData) -> Self {
        Self { data }
    }
}

impl ForwardLink<NodeLocation> for PartitionForwardLink {
    fn node(self) -> NodeLocation {
        unsafe {
            // Safety: ForwardLink is constructed using a NodeLocation
            let partition = (self.inner() >> 25) as usize;
            let index = ((self.inner() >> 5) & 0b11111111111111111111) as usize;
            NodeLocation::from(partition, index)
        }
    }

    fn side(self) -> bool {
        self.inner() & (1 << 4) != 0
    }

    fn ss(self) -> u8 {
        (self.inner() & 0b1111) as u8
    }
}
