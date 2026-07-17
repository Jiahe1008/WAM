import torch


PUSHER_RADIUS = 0.06
OBJECT_RADIUS = 0.08
MIN_CONTACT_DIST = PUSHER_RADIUS + OBJECT_RADIUS
CONTACT_MARGIN = 0.01
ENV_DT = 0.05
GRAVITY = 9.81
EPS = 1e-6


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_physics_loss(norm_states, norm_pred_next_states, state_mean, state_std):
    """
    Physics-informed constraints for WAM world-head prediction.

    State layout:
        0:2   pusher xy
        2:4   object xy
        4:6   object velocity xy
        6:8   target xy
        8     object mass
        9     object friction

    Loss terms:
        1. No-contact no-source energy: without contact, object kinetic energy
           should not increase.
        2. Contact impulse direction: under contact, object velocity change should
           be consistent with the pusher-to-object contact normal.
        3. Non-penetration: pusher/object rigid bodies should not overlap.
    """
    states = norm_states * state_std + state_mean
    pred_next_states = norm_pred_next_states * state_std + state_mean

    pusher_xy = states[:, 0:2]
    object_xy = states[:, 2:4]
    cur_v = states[:, 4:6]
    mass = states[:, 8].clamp_min(EPS)

    pred_pusher_xy = pred_next_states[:, 0:2]
    pred_object_xy = pred_next_states[:, 2:4]
    pred_v = pred_next_states[:, 4:6]

    cur_dist = torch.norm(object_xy - pusher_xy, dim=-1)
    pred_dist = torch.norm(pred_object_xy - pred_pusher_xy, dim=-1)
    cur_gap = cur_dist - MIN_CONTACT_DIST
    pred_gap = pred_dist - MIN_CONTACT_DIST

    # 1. Non-contact states should not gain kinetic energy without contact work.
    no_contact_mask = (cur_gap > CONTACT_MARGIN) & (pred_gap > CONTACT_MARGIN)
    cur_kinetic = 0.5 * mass * torch.sum(cur_v ** 2, dim=-1)
    pred_kinetic = 0.5 * mass * torch.sum(pred_v ** 2, dim=-1)
    loss_no_contact_energy = _masked_mean(
        torch.relu(pred_kinetic - cur_kinetic) ** 2,
        no_contact_mask,
    )

    # 2. Contact impulse should push the object away from the pusher.
    contact_mask = (cur_gap <= CONTACT_MARGIN) | (pred_gap <= CONTACT_MARGIN)
    contact_normal = object_xy - pusher_xy
    contact_normal = contact_normal / (
        torch.norm(contact_normal, dim=-1, keepdim=True).clamp_min(EPS)
    )
    delta_v = pred_v - cur_v
    normal_impulse_sign = torch.sum(delta_v * contact_normal, dim=-1)
    loss_impulse_direction = _masked_mean(
        torch.relu(-normal_impulse_sign) ** 2,
        contact_mask,
    )

    # 3. Rigid bodies should not overlap.
    loss_nonpenetration = torch.mean(torch.relu(-pred_gap) ** 2)

    return (
        loss_nonpenetration
        + loss_no_contact_energy
        + loss_impulse_direction
    )


def compute_dynamics_residual_loss(
    norm_states,
    norm_next_states,
    pred_contact_force,
    state_mean,
    state_std,
    dt=ENV_DT,
    gravity=GRAVITY,
):
    """
    Self-supervised force consistency loss.

    The model predicts contact force; true velocity change from data supplies the
    Newton residual:

        m * (v_next - v_t) / dt = F_contact_pred + F_friction

    No MuJoCo contact-force labels are required. This constrains the learned force
    to explain observed object acceleration.
    """
    states = norm_states * state_std + state_mean
    next_states = norm_next_states * state_std + state_mean

    pusher_xy = states[:, 0:2]
    object_xy = states[:, 2:4]
    cur_v = states[:, 4:6]
    next_v = next_states[:, 4:6]
    mass = states[:, 8].clamp_min(EPS)
    friction = states[:, 9].clamp_min(0.0)

    acceleration = (next_v - cur_v) / dt
    required_force = mass.unsqueeze(-1) * acceleration

    speed = torch.norm(cur_v, dim=-1, keepdim=True)
    moving_mask = (speed > 1e-4).to(dtype=cur_v.dtype)
    velocity_dir = cur_v / speed.clamp_min(EPS)
    friction_force = (
        -friction.unsqueeze(-1)
        * mass.unsqueeze(-1)
        * gravity
        * velocity_dir
        * moving_mask
    )

    residual = required_force - (pred_contact_force + friction_force)
    loss_dynamics_residual = torch.mean(torch.sum(residual ** 2, dim=-1))

    # Without contact, the learned contact force should stay small. Kept light
    # because MuJoCo friction/contact details are richer than this simple model.
    cur_gap = torch.norm(object_xy - pusher_xy, dim=-1) - MIN_CONTACT_DIST
    no_contact_mask = cur_gap > CONTACT_MARGIN
    loss_no_contact_force = _masked_mean(
        torch.sum(pred_contact_force ** 2, dim=-1),
        no_contact_mask,
    )

    return loss_dynamics_residual + 0.01 * loss_no_contact_force
