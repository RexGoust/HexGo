use bevy::prelude::*;

use crate::client::SessionResource;
use crate::client::materials::StoneMaterials;
use crate::client::state::InGameEntity;
use crate::game::board::VertexId;

pub const BOARD_BACKGROUND: Color = Color::srgb(0.84, 0.69, 0.39);
const LINE_COLOR: Color = Color::srgb(0.22, 0.16, 0.10);
const FOCUS_MARKER_COLOR: Color = Color::srgba(0.96, 0.63, 0.22, 0.78);
const LAST_MOVE_MARKER_COLOR: Color = Color::srgb(0.78, 0.30, 0.21);

#[derive(Component)]
pub enum Marker {
    Focus,
    LastMove,
}

#[derive(Component)]
pub struct BoardRoot;

#[derive(Component)]
pub struct Stone(pub(crate) VertexId);

#[derive(Component)]
pub struct PreviewStone;

pub fn setup_board(
    mut commands: Commands,
    session: Res<SessionResource>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    stone_materials: Res<StoneMaterials>,
) {
    let root = commands
        .spawn((
            BoardRoot,
            InGameEntity,
            Transform::default(),
            Visibility::Visible,
        ))
        .id();

    commands.entity(root).with_children(|parent| {
        spawn_board_lines(parent, &session);
        spawn_stones(parent, &session, &mut meshes, &stone_materials);
        spawn_markers(parent, &mut meshes, &mut materials);
    });
}

fn spawn_board_lines(parent: &mut ChildSpawnerCommands, session: &SessionResource) {
    for &(left, right) in session.0.definition().edges() {
        let left = Vec2::from_array(session.0.definition().position(left).unwrap());

        let right = Vec2::from_array(session.0.definition().position(right).unwrap());

        let delta = right - left;

        parent.spawn((
            Sprite::from_color(LINE_COLOR, Vec2::new(delta.length() + 0.025, 0.055)),
            Transform::from_translation(((left + right) * 0.5).extend(0.0))
                .with_rotation(Quat::from_rotation_z(delta.y.atan2(delta.x))),
        ));
    }
}

fn spawn_stones(
    parent: &mut ChildSpawnerCommands,
    session: &SessionResource,
    meshes: &mut Assets<Mesh>,
    stone_materials: &StoneMaterials,
) {
    let stone_mesh = meshes.add(Circle::new(0.34));

    for vertex in session.0.definition().graph().vertices() {
        let position = Vec2::from_array(session.0.definition().position(vertex).unwrap());

        parent.spawn((
            Stone(vertex),
            Mesh2d(stone_mesh.clone()),
            MeshMaterial2d(stone_materials.black.clone()),
            Transform::from_translation(position.extend(2.0)),
            Visibility::Hidden,
        ));
    }

    parent.spawn((
        PreviewStone,
        Mesh2d(stone_mesh),
        MeshMaterial2d(stone_materials.preview_black.clone()),
        Transform::from_xyz(0.0, 0.0, 3.0),
        Visibility::Hidden,
    ));
}

fn spawn_markers(
    parent: &mut ChildSpawnerCommands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<ColorMaterial>,
) {
    parent.spawn((
        Marker::Focus,
        Mesh2d(meshes.add(Circle::new(0.43))),
        MeshMaterial2d(materials.add(FOCUS_MARKER_COLOR)),
        Transform::from_xyz(0.0, 0.0, 1.0),
        Visibility::Hidden,
    ));

    parent.spawn((
        Marker::LastMove,
        Mesh2d(meshes.add(Circle::new(0.09))),
        MeshMaterial2d(materials.add(LAST_MOVE_MARKER_COLOR)),
        Transform::from_xyz(0.0, 0.0, 4.0),
        Visibility::Hidden,
    ));
}
