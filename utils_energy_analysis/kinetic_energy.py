

def compute_ke(uvel, vvel, wvel, dx, dy, dz, rho=1000.0):
    ke = 0.5 * rho * (uvel ** 2 + vvel ** 2 + wvel ** 2) * dx * dy * dz  # This gives J per cell

    return ke / 1e6  # Convert to MJ