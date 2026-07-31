"""Building the world: the ground, the track, the walls and the cars.

USD plumbing, run once at start-up. Split out from :mod:`lituanicax_sdk.env`
because it has nothing to do with the simulation loop — it is the one-time job
of turning meshes and config into a physics scene, and reading it is rarely
what you want when you are looking at how a step works.

``omni`` and ``pxr`` only exist once the Isaac Sim app is running, which is why
every function here imports them inside its own body.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from .track import Track
from .vehicle import VEHICLE, build_ground_material_cfg


def spawn_track(track: Track) -> None:
    """Place the track surface and the walls. One shared copy of each.

    The surface is scenery — cones, tarmac, kerbs — and is never given
    collision, so decorating a track cannot change how it drives. The walls
    are a physical boundary: exact collision, kinematic, and invisible.
    """
    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import Usd, UsdGeom, UsdPhysics  # pyright: ignore[reportMissingImports]

    cfg = track.cfg
    scale = (cfg.mesh_scale, cfg.mesh_scale, cfg.mesh_scale)

    if cfg.surface_usd:
        surface = sim_utils.UsdFileCfg(usd_path=cfg.surface_usd, scale=scale)
        surface.func(
            track.surface_prim_path,
            surface,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

    walls = sim_utils.UsdFileCfg(usd_path=cfg.walls_usd, scale=scale)
    walls.func(
        track.walls_prim_path,
        walls,
        translation=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    stage = omni.usd.get_context().get_stage()
    walls_root = stage.GetPrimAtPath(track.walls_prim_path)
    for prim in Usd.PrimRange(walls_root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim).GetCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("none")

    body = UsdPhysics.RigidBodyAPI.Apply(walls_root)
    body.GetRigidBodyEnabledAttr().Set(True)
    body.GetKinematicEnabledAttr().Set(True)
    UsdGeom.Imageable(walls_root).MakeInvisible()


def spawn_ground() -> None:
    """The ground plane, with the locked tyre-to-tarmac friction."""
    ground = sim_utils.GroundPlaneCfg(physics_material=build_ground_material_cfg())
    ground.func("/World/GroundPlane", ground)


def spawn_light(intensity: float) -> None:
    light = sim_utils.DomeLightCfg(intensity=intensity, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light)


def prepare_car_physics(num_envs: int) -> None:
    """Wheel collision hulls, and clearing the USD's own suspension springs."""
    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import Usd, UsdGeom, UsdPhysics  # pyright: ignore[reportMissingImports]

    stage = omni.usd.get_context().get_stage()
    wheel_links = set(VEHICLE.wheel_links)

    car_root = stage.GetPrimAtPath(
        f"/World/envs/env_0/Robot/{VEHICLE.articulation_subpath}"
    )
    if car_root.IsValid():
        for prim in Usd.PrimRange(car_root):
            if _is_wheel_mesh(prim, UsdGeom, wheel_links):
                # A cheap convex hull is plenty for a wheel.
                UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set(
                    "convexHull"
                )

    # The USD ships its own suspension spring values; clear them so the locked
    # actuator settings are the only thing in charge.
    for env_id in range(num_envs):
        base = (
            f"/World/envs/env_{env_id}/Robot/"
            f"{VEHICLE.articulation_subpath}/base_link"
        )
        for joint_name in VEHICLE.suspension_joints:
            joint = stage.GetPrimAtPath(f"{base}/{joint_name}")
            if not joint.IsValid():
                continue
            for prop in list(joint.GetProperties()):
                name = prop.GetName()
                if "stiffness" in name or "damping" in name:
                    attribute = joint.GetAttribute(name)
                    if attribute.IsValid():
                        attribute.Block()


def define_tyre_material():
    """Create the rubber physics material. Bind it *after* cloning."""
    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import UsdPhysics  # pyright: ignore[reportMissingImports]

    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim("/World/PhysicsMaterials/WheelRubber", "Material")
    UsdPhysics.MaterialAPI.Apply(prim)
    api = UsdPhysics.MaterialAPI(prim)
    api.CreateStaticFrictionAttr().Set(VEHICLE.tyre_static_friction)
    api.CreateDynamicFrictionAttr().Set(VEHICLE.tyre_dynamic_friction)
    api.CreateRestitutionAttr().Set(VEHICLE.tyre_restitution)
    return prim


def bind_tyre_material(num_envs: int, rubber_prim) -> None:
    """Bind the tyre material to every wheel of every cloned car."""
    import omni.usd  # pyright: ignore[reportMissingImports]
    from pxr import Usd, UsdGeom, UsdShade  # pyright: ignore[reportMissingImports]

    stage = omni.usd.get_context().get_stage()
    material = UsdShade.Material(rubber_prim)
    wheel_links = set(VEHICLE.wheel_links)

    for env_id in range(num_envs):
        root = stage.GetPrimAtPath(
            f"/World/envs/env_{env_id}/Robot/{VEHICLE.articulation_subpath}"
        )
        if not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if _is_wheel_mesh(prim, UsdGeom, wheel_links):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material, UsdShade.Tokens.strongerThanDescendants, "physics"
                )


def spawn_extra_entity(scene, name: str, entity_cfg) -> None:
    """Add one of the team's own scene entities.

    Handles the three Isaac Lab kinds you are likely to want: sensors, rigid
    objects and articulations. Anything else with a ``func`` is spawned as
    plain USD.
    """
    from isaaclab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from isaaclab.sensors import SensorBaseCfg

    if isinstance(entity_cfg, SensorBaseCfg):
        scene.sensors[name] = entity_cfg.class_type(entity_cfg)
    elif isinstance(entity_cfg, RigidObjectCfg):
        scene.rigid_objects[name] = RigidObject(entity_cfg)
    elif isinstance(entity_cfg, ArticulationCfg):
        scene.articulations[name] = Articulation(entity_cfg)
    elif hasattr(entity_cfg, "func"):
        entity_cfg.func(f"/World/extras/{name}", entity_cfg)
    else:
        raise TypeError(
            f"extra_scene_entities['{name}'] is a {type(entity_cfg).__name__}, "
            "which the SDK does not know how to spawn. Use a sensor, rigid "
            "object or articulation config, or a spawner config with .func()."
        )


def _is_wheel_mesh(prim, usd_geom, wheel_links: set[str]) -> bool:
    """True if ``prim`` is one of the four wheel meshes."""
    return prim.IsA(usd_geom.Mesh) and prim.GetParent().GetName() in wheel_links
