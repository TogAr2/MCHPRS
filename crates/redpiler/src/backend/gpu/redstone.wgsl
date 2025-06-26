const _REPEATER = 0u;
const _TORCH = 1u;
const _COMPARATOR = 2u;
const _LAMP = 3u;
const _BUTTON = 4u;
const _LEVER = 5u;
const _PRESSURE_PLATE = 6u;
const _TRAPDOOR = 7u;
const _WIRE = 8u;
const _CONSTANT = 9u;
const _NOTE_BLOCK = 10u;

struct TickPriority {
    data: u32,
};

const _HIGHEST = 0u;
const _HIGHER = 1u;
const _HIGH = 2u;
const _NORMAL = 3u;

struct NeighborInfo {
    count: u32,
    index: u32,
};

struct ForwardLink {
    distance: u32,
    is_side: bool,
    node_index: u32,
};

struct TypeInfo {
    ty: u32,
    data: u32, // Repeater: delay, Comparator: mode, Note block: id
    facing_diode: bool,
    far_input: u32, // Will be 255 for no input
};

struct State {
    output_strength: u32,
    repeater_locked: bool,
    changed: bool,
};

fn decode_neighbor_info(packed: u32) -> NeighborInfo {
    return NeighborInfo(
        packed & 0xFFu,
        packed >> 8u,
    );
}

fn decode_link(packed: u32) -> ForwardLink {
    return ForwardLink(
        packed & 0xFu,
        ((packed >> 4u) & 1u) != 0,
        packed >> 5u
    );
}

fn decode_type(packed: u32) -> TypeInfo {
    return TypeInfo(
        packed & 0xFFu,
        (packed >> 8u) & 0xFFu,
        ((packed >> 16u) & 1u) != 0u,
        (packed >> 17u) & 0xFFu,
    );
}

fn encode_state(state: State) -> u32 {
    let output = (state.output_strength & 0xFFu);
    let locked = select(0u, 1u, state.repeater_locked) << 8u;
    let changed = select(0u, 1u, state.changed) << 9u;
    return output | locked | changed;
}

fn decode_state(packed: u32) -> State {
    return State(
        packed & 0xFFu,
        ((packed >> 8u) & 1u) != 0u,
        ((packed >> 9u) & 1u) != 0u,
    );
}

// GROUPS:
// 0 - Constants
// 1 - Common
// 2 - Only tick invocation

// Constant
@group(0) @binding(0) var<storage, read> types: array<u32>;
@group(0) @binding(1) var<storage, read> neighbor_info: array<u32>;
@group(0) @binding(2) var<storage, read> neighbor_links: array<u32>;


// Common: States -- invocation will only access its own state so not double-buffered
@group(1) @binding(0) var<storage, read_write> states: array<u32>;

// Common: Input strengths in -- only read
@group(1) @binding(1) var<storage, read> default_inputs_in: array<array<u32, 16>>;
@group(1) @binding(2) var<storage, read> side_inputs_in: array<array<u32, 16>>;

// Only tick: Input strengths out -- invocation might access other inputs, so atomic is used
@group(2) @binding(0) var<storage, read_write> default_inputs_out: array<array<atomic<u32>, 16>>;
@group(2) @binding(1) var<storage, read_write> side_inputs_out: array<array<atomic<u32>, 16>>;

// Only tick: Current tick priority
@group(2) @binding(2) var<uniform> tick_priority: TickPriority;


// Common: Tick delay in -- only read
@group(1) @binding(3) var<storage, read> tick_delay_in: array<u32>;

// Common: Tick delay out -- invocation might access other tick delays, so atomic is used
@group(1) @binding(4) var<storage, read_write> tick_delay_out: array<atomic<u32>>;

@compute @workgroup_size(32)
fn update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Avoid out of bound access (if more invocations than nodes)
    let array_length = arrayLength(&types);
    if index >= array_length {
        return;
    }

    let state_raw = states[index];
    if state_raw == 0xFFFFFFFFu {
        // This data should not be operated on
        return;
    }
    var state = decode_state(states[index]);

    let input_strength = get_max_ss(default_inputs_in[index]);
    let side_input_strength = get_max_ss(side_inputs_in[index]);
    let powered = state.output_strength > 0u;

    let pending_tick =
        tick_delay_in[index] > 0u ||
        tick_delay_in[array_length + index] > 0u ||
        tick_delay_in[2 * array_length + index] > 0u ||
        tick_delay_in[3 * array_length + index] > 0u;

    let type_info = decode_type(types[index]);

    switch (type_info.ty) {
        case _REPEATER: {
            let should_be_locked = side_input_strength > 0u;
            if should_be_locked != state.repeater_locked {
                state.repeater_locked = should_be_locked;
                state.changed = true;
            }
            if !state.repeater_locked && !pending_tick {
                let should_be_powered = input_strength > 0u;
                if should_be_powered != powered {
                    var priority: u32;
                    if type_info.facing_diode {
                        priority = _HIGHEST;
                    } else if !should_be_powered {
                        priority = _HIGHER;
                    } else {
                        priority = _HIGH;
                    }
                    let delay = type_info.data;
                    schedule_tick(index, delay, priority);
                }
            }
        }
        case _TORCH: {
            if !pending_tick {
                let should_be_powered = input_strength == 0u;
                if powered != should_be_powered {
                    schedule_tick(index, 1u, _NORMAL);
                }
            }
        }
        case _COMPARATOR: {
            if !pending_tick {
                var actual_input: u32 = input_strength;
                if input_strength < 15u && type_info.far_input != 255u {
                    actual_input = type_info.far_input;
                }
                let subtract = type_info.data == 1u;
                let new_strength = get_comparator_output(subtract, actual_input, side_input_strength);
                if new_strength != state.output_strength {
                    var priority: u32;
                    if type_info.facing_diode {
                        priority = _HIGH;
                    } else {
                        priority = _NORMAL;
                    }
                    schedule_tick(index, 1u, priority);
                }
            }
        }
        case _LAMP: {
            let should_be_powered = input_strength > 0u;
            if powered && !should_be_powered {
                schedule_tick(index, 2u, _NORMAL);
            } else if !powered && should_be_powered {
                // Output strength will never be used, this is purely an indicator the lamp is on
                state.output_strength = 15;
                state.changed = true;
            }
        }
        case _TRAPDOOR: {
            let should_be_powered = input_strength > 0u;
            if powered != should_be_powered {
                state.output_strength = bool_to_ss(should_be_powered);
                state.changed = true;
            }
        }
        case _WIRE: {
            if state.output_strength != input_strength {
                state.output_strength = input_strength;
                state.changed = true;
            }
        }
        case _NOTE_BLOCK: {
            let should_be_powered = input_strength > 0u;
            if powered != should_be_powered {
                state.output_strength = bool_to_ss(should_be_powered);
                if should_be_powered {
                    //TODO note block play!
                }
            }
        }
        default: {}
    }

    states[index] = encode_state(state);
}

@compute @workgroup_size(32)
fn tick(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Avoid out of bound access (if more invocations than nodes)
    let array_length = arrayLength(&types);
    if index >= array_length {
        return;
    }

    let state_raw = states[index];
    if state_raw == 0xFFFFFFFFu {
        // This data should not be used
        return;
    }

    let tick_index = tick_priority.data * array_length + index;
    let tick_delay = tick_delay_in[tick_index];
    if tick_delay == 0u {
        return; // Node has no pending tick
    }

    let new_tick_delay = tick_delay - 1u;
    tick_delay_out[tick_index] = new_tick_delay;
    if new_tick_delay != 0u {
        return; // Doesn't need to be ticked yet
    }

    var state = decode_state(states[index]);
    if state.repeater_locked {
        return;
    }

    let input_strength = get_max_ss(default_inputs_in[index]);
    let side_input_strength = get_max_ss(side_inputs_in[index]);
    let powered = state.output_strength > 0u;

    let type_info = decode_type(types[index]);
    let n_info = decode_neighbor_info(neighbor_info[index]);

    switch (type_info.ty) {
        case _REPEATER: {
            let should_be_powered = input_strength > 0u;
            if powered && !should_be_powered {
                update_state(&state, 0u, n_info);
            } else if !powered {
                if !should_be_powered {
                    let delay = type_info.data;
                    //TODO in principle this is the only output tick priority being used in the tick pass, so maybe optimize memory usage
                    schedule_tick(index, delay, _HIGHER);
                }
                update_state(&state, 15u, n_info);
            }
        }
        case _TORCH: {
            let should_be_powered = input_strength == 0u;
            if powered != should_be_powered {
                update_state(&state, bool_to_ss(should_be_powered), n_info);
            }
        }
        case _COMPARATOR: {
            var actual_input: u32 = input_strength;
            if input_strength < 15u && type_info.far_input != 255u {
                actual_input = type_info.far_input;
            }
            let subtract = type_info.data == 1u;
            let new_strength = get_comparator_output(subtract, actual_input, side_input_strength);
            if new_strength != state.output_strength {
                update_state(&state, new_strength, n_info);
            }
        }
        case _LAMP: {
            let should_be_powered = input_strength > 0u;
            if powered && !should_be_powered {
                update_state(&state, 0u, n_info);
            }
        }
        case _BUTTON: {
            if powered {
                update_state(&state, 0u, n_info);
            }
        }
        default: {}
    };

    states[index] = encode_state(state);
}

fn update_state(state: ptr<function, State>, new_out_strength: u32, neighbor_info: NeighborInfo) {
    let prev_out_strength = state.output_strength;
    (*state).output_strength = new_out_strength;
    (*state).changed = true;

    let min = neighbor_info.index;
    let max = min + neighbor_info.count;
    for (var i: u32 = min; i < max; i++) {
        let link = decode_link(neighbor_links[i]);

        let prev_strength = prev_out_strength - link.distance;
        let new_strength = new_out_strength - link.distance;

        if link.is_side {
            atomicSub(&side_inputs_out[link.node_index][prev_strength], 1u);
            atomicAdd(&side_inputs_out[link.node_index][new_strength], 1u);
        } else {
            atomicSub(&default_inputs_out[link.node_index][prev_strength], 1u);
            atomicAdd(&default_inputs_out[link.node_index][new_strength], 1u);
        }
    }
}

fn schedule_tick(index: u32, delay: u32, priority: u32) {
    //TODO check that maxing delay actually works
    let array_length = arrayLength(&types);
    atomicMax(&tick_delay_out[priority * array_length + index], delay);
}

fn get_max_ss(inputs: array<u32, 16>) -> u32 {
    var max_val: u32 = inputs[0];
    var max_idx: u32 = 0u;

    for (var i: u32 = 0u; i < 16u; i++) {
        let val = inputs[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    return max_idx;
}

fn bool_to_ss(powered: bool) -> u32 {
    if powered {
        return 15u;
    } else {
        return 0u;
    }
}

fn get_comparator_output(subtract: bool, input_strength: u32, side_input_strength: u32) -> u32 {
    if subtract {
        if input_strength > side_input_strength {
            return input_strength - side_input_strength;
        } else {
            return 0u;
        }
    } else {
        if input_strength >= side_input_strength {
            return input_strength;
        } else {
            return 0u;
        }
    }
}
