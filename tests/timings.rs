mod common;
use common::*;

use mchprs_blocks::blocks::{Block, ComparatorMode};
use mchprs_blocks::BlockDirection;
use mchprs_world::World;

test_all_backends!(repeater_t_flip_flop, optimize=false);
fn repeater_t_flip_flop(backend: TestBackend) {
    // RN -> Repeater North
    // Layout:
    // W RN W
    // W RN RE
    // L

    let mut world = TestWorld::new(1);

    let output_pos = pos(1, 1, 2);
    let lever_pos = pos(0, 1, 0);

    make_lever(&mut world, lever_pos);
    make_wire(&mut world, pos(1, 1, 0));
    make_wire(&mut world, pos(2, 1, 0));

    make_repeater(&mut world, pos(1, 1, 1), 1, BlockDirection::North, false);
    make_repeater(&mut world, pos(2, 1, 1), 1, BlockDirection::North, false);

    make_repeater(&mut world, output_pos, 1, BlockDirection::East, false);
    make_wire(&mut world, pos(2, 1, 2));

    let mut runner = BackendRunner::new(world, backend);
    // Set up initial state
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 2);

    // Toggle flip flop on
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 2);
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, true, 10);

    // Toggle flip flop off
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, true, 2);
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 10);
}

test_all_backends!(pulse_gen_2t, optimize=true);
fn pulse_gen_2t(backend: TestBackend) {
    let output_pos = pos(4, 1, 1);
    let lever_pos = pos(0, 1, 1);

    let mut world = TestWorld::new(1);

    make_wire(&mut world, pos(1, 1, 0));
    make_repeater(&mut world, pos(2, 1, 0), 2, BlockDirection::West, false);
    make_wire(&mut world, pos(3, 1, 0));

    make_lever(&mut world, lever_pos);
    make_wire(&mut world, pos(1, 1, 1));
    make_wire(&mut world, pos(2, 1, 1));
    make_comparator(
        &mut world,
        pos(3, 1, 1),
        ComparatorMode::Subtract,
        BlockDirection::West,
    );
    place_on_block(&mut world, output_pos, trapdoor());

    let mut runner = BackendRunner::new(world, backend);

    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 1);
    runner.check_powered_for(output_pos, true, 2);
    runner.check_powered_for(output_pos, false, 10);
}

test_all_backends!(pulse_gen_1t, optimize=true);
fn pulse_gen_1t(backend: TestBackend) {
    let output_pos = pos(5, 1, 1);
    let lever_pos = pos(0, 1, 1);

    let mut world = TestWorld::new(1);

    make_wire(&mut world, pos(1, 1, 0));
    make_repeater(&mut world, pos(2, 1, 0), 2, BlockDirection::West, false);
    make_wire(&mut world, pos(3, 1, 0));
    make_wire(&mut world, pos(4, 1, 0));

    make_lever(&mut world, lever_pos);
    make_wire(&mut world, pos(1, 1, 1));
    make_wire(&mut world, pos(2, 1, 1));
    make_comparator(
        &mut world,
        pos(3, 1, 1),
        ComparatorMode::Subtract,
        BlockDirection::West,
    );
    place_on_block(&mut world, pos(4, 1, 1), Block::Sandstone {});
    place_on_block(&mut world, output_pos, trapdoor());

    let mut runner = BackendRunner::new(world, backend);

    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 1);
    runner.check_powered_for(output_pos, true, 1);
    runner.check_powered_for(output_pos, false, 10);
}

test_all_backends!(pulse_length_convert, optimize=true);
fn pulse_length_convert(backend: TestBackend) {
    let output_pos = pos(5, 1, 0);
    let lever_pos = pos(0, 1, 0);

    let mut world = TestWorld::new(1);

    make_lever(&mut world, lever_pos);
    make_repeater(&mut world, pos(1, 1, 0), 4, BlockDirection::West, false);
    make_repeater(&mut world, pos(2, 1, 0), 4, BlockDirection::West, false);
    make_repeater(&mut world, pos(3, 1, 0), 2, BlockDirection::West, false);
    make_repeater(&mut world, pos(4, 1, 0), 4, BlockDirection::West, false);
    place_on_block(&mut world, output_pos, trapdoor());

    let mut runner = BackendRunner::new(world, backend);

    runner.use_block(lever_pos);
    runner.tick();
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 13);
    runner.check_powered_for(output_pos, true, 4);
    runner.check_powered_for(output_pos, false, 10);
}

test_all_backends!(torch_chain_1t_pulse, optimize=true);
fn torch_chain_1t_pulse(backend: TestBackend) {
    let output_pos_1 = pos(4, 2, 0);
    let output_pos_2 = pos(7, 1, 0);
    let lever_pos = pos(0, 1, 0);

    let mut world = TestWorld::new(1);

    make_lever(&mut world, lever_pos);
    make_repeater(&mut world, pos(1, 1, 0), 1, BlockDirection::West, false);
    make_repeater(&mut world, pos(2, 1, 0), 1, BlockDirection::West, false);
    make_repeater(&mut world, pos(3, 1, 0), 1, BlockDirection::West, false);
    world.set_block(pos(4, 1, 0), Block::Sandstone {});
    world.set_block(output_pos_1, trapdoor());
    world.set_block(
        pos(5, 1, 0),
        Block::RedstoneWallTorch {
            lit: true,
            facing: BlockDirection::West,
        },
    );
    make_repeater(&mut world, pos(6, 1, 0), 1, BlockDirection::West, false);
    place_on_block(&mut world, output_pos_2, trapdoor());

    let mut runner = BackendRunner::new(world, backend);

    runner.use_block(lever_pos);
    runner.tick();
    runner.use_block(lever_pos);
    runner.check_block_powered(output_pos_2, false);
    runner.check_powered_for(output_pos_1, false, 2);
    runner.check_block_powered(output_pos_2, false);
    runner.check_powered_for(output_pos_1, true, 1);
    runner.check_block_powered(output_pos_1, false);
    runner.check_powered_for(output_pos_2, false, 10);
}

test_all_backends!(repeated_inversion_chain, optimize=true);
fn repeated_inversion_chain(backend: TestBackend) {
    let output_pos = pos(7, 1, 0);
    let lever_pos = pos(0, 1, 0);

    let mut world = TestWorld::new(1);

    make_lever(&mut world, lever_pos);
    make_repeater(&mut world, pos(1, 1, 0), 1, BlockDirection::West, false);
    world.set_block(pos(2, 1, 0), Block::Sandstone {});
    world.set_block(
        pos(3, 1, 0),
        Block::RedstoneWallTorch {
            lit: true,
            facing: BlockDirection::East,
        },
    );
    make_repeater(&mut world, pos(4, 1, 0), 1, BlockDirection::West, true);
    world.set_block(pos(5, 1, 0), Block::Sandstone {});
    world.set_block(
        pos(6, 1, 0),
        Block::RedstoneWallTorch {
            lit: false,
            facing: BlockDirection::East,
        },
    );
    place_on_block(&mut world, output_pos, trapdoor());

    let mut runner = BackendRunner::new(world, backend);

    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 2);
    runner.use_block(lever_pos);
    runner.check_powered_for(output_pos, false, 2);
    runner.check_powered_for(output_pos, true, 2);
    runner.check_powered_for(output_pos, false, 10);
}
