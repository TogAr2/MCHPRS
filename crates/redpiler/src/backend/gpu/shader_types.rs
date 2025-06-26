#[repr(u32)]
pub enum TickPriority {
    Highest = 0,
    Higher = 1,
    High = 2,
    Normal = 3,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum NodeType {
    Repeater = 0,
    Torch = 1,
    Comparator = 2,
    Lamp = 3,
    Button = 4,
    Lever = 5,
    PressurePlate = 6,
    Trapdoor = 7,
    Wire = 8,
    Constant = 9,
    NoteBlock = 10,
}

#[derive(Debug)]
pub struct NeighborInfo {
    pub count: u8,
    pub index: usize,
}

impl NeighborInfo {
    pub fn as_packed(&self) -> u32 {
        let count = self.count as u32;
        let index = (self.index as u32) << 8;
        count | index
    }
}

#[derive(Debug)]
pub struct ForwardLink {
    pub distance: u8,
    pub is_side: bool,
    pub node_index: usize,
}

impl ForwardLink {
    pub fn as_packed(&self) -> u32 {
        let distance = self.distance as u32;
        let is_side = (self.is_side as u32) << 4;
        let node_index = (self.node_index as u32) << 5;
        distance | is_side | node_index
    }
}

#[derive(Debug)]
pub struct TypeInfo {
    pub ty: NodeType,
    /// Repeater: delay, Comparator: 1 if subtract, Note block: id
    pub data: u8,
    pub facing_diode: bool,
    /// Will be 255 for no input
    pub far_input: u8,
}

impl TypeInfo {
    pub fn as_packed(&self) -> u32 {
        let ty = self.ty as u32;
        let data = (self.data as u32) << 8;
        let facing = (self.facing_diode as u32) << 16;
        let far_input = (self.far_input as u32) << 17;
        ty | data | facing | far_input
    }
}

#[derive(Debug)]
pub struct State {
    pub output_strength: u8,
    pub repeater_locked: bool,
    pub changed: bool,
}

impl State {
    pub fn as_packed(&self) -> u32 {
        let output = self.output_strength as u32;
        let repeater_locked = (self.repeater_locked as u32) << 8;
        let changed = (self.changed as u32) << 9;
        output | repeater_locked | changed
    }

    pub fn from_packed(packed: u32) -> State {
        Self {
            output_strength: (packed & 0xFF) as u8,
            repeater_locked: (packed >> 8) & 0x1 != 0,
            changed: (packed >> 9) & 0x1 != 0,
        }
    }
}
