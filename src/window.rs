use bevy::{
    utils::default,
    window::{Window, WindowResizeConstraints, WindowResolution},
};

pub fn primary_window() -> Window {
    let resize_constraints = if cfg!(any(target_os = "android", target_arch = "wasm32")) {
        // Embedded surfaces follow the available viewport, including small mobile screens.
        WindowResizeConstraints {
            min_width: 1.0,
            min_height: 1.0,
            ..default()
        }
    } else {
        WindowResizeConstraints {
            min_width: 360.0,
            min_height: 480.0,
            ..default()
        }
    };

    Window {
        title: "HexGo".into(),
        resolution: WindowResolution::new(1280, 800),
        resize_constraints,
        resizable: !cfg!(target_os = "android"),
        canvas: Some("#hexgo-canvas".into()),
        fit_canvas_to_parent: true,
        ..default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_canvas_tracks_its_container() {
        let window = primary_window();
        assert_eq!(window.canvas.as_deref(), Some("#hexgo-canvas"));
        assert!(window.fit_canvas_to_parent);
        let page = include_str!("../index.html");
        assert!(page.contains("id=\"hexgo-canvas\""));
        assert!(page.contains("id=\"loading-screen\""));
        assert!(page.contains("data-initializer=\"web/loader.mjs\""));
        assert!(page.contains("<base data-trunk-public-url />"));
        let trunk_config = include_str!("../Trunk.toml");
        assert!(trunk_config.contains("public_url = \"./\""));
    }
}
