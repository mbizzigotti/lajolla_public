
/// Given a vector n, outputs two vectors such that all three vectors are
/// orthogonal to each other.
/// The approach here is based on Frisvad's paper
/// "Building an Orthonormal Basis from a 3D Unit Vector Without Normalization"
/// https://backend.orbit.dtu.dk/ws/portalfiles/portal/126824972/onb_frisvad_jgt2012_v2.pdf
void coordinate_system(in float3 n, out float3 x, out float3 y)
{
    if (n[2] < float(-1 + 1e-6)) {
        x = float3(0, -1, 0);
        y = float3(-1, 0, 0);
    } else {
        float a = 1 / (1 + n[2]);
        float b = -n[0] * n[1] * a;
        x = float3(1 - n[0] * n[0] * a, b, -n[0]);
        y = float3(b, 1 - n[1] * n[1] * a, -n[1]);
    }
}

/// A "Frame" is a coordinate basis that consists of three orthogonal unit vectors.
/// This is useful for sampling points on a hemisphere or defining anisotropic BSDFs.
struct Frame
{
    float3 x, y, n; // 0, 1, 2
    
    __init()
    {
        x = 0;
        y = 0;
        n = 0;
    }
    __init(float3 x, float3 y, float3 n)
    {
        this.x = x;
        this.y = y;
        this.n = n;
    }

    __init(float3 n)
    {
        this.n = n;
        coordinate_system(n, x, y);
    }
};

__prefix Frame operator -(Frame frame)
{
    return { -frame.x, -frame.y, -frame.n };
}

float3 to_world(in Frame frame, float3 v)
{
    return frame.x * v.x + frame.y * v.y + frame.n * v.z;
}

/// Project a vector to a frame's local coordinates.
float3 to_local(in Frame frame, float3 v)
{
    return float3(dot(v, frame.x), dot(v, frame.y), dot(v, frame.n));
}

/// See "Sampling the GGX Distribution of Visible Normals", Heitz, 2018.
/// https://jcgt.org/published/0007/04/01/
[ForceInline]
float3 sample_visible_normals(float3 local_dir_in, float alpha, float2 rnd_param)
{
    float factor = 1;
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
        factor = -1;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(
        float3(alpha * local_dir_in.x, alpha * local_dir_in.y, local_dir_in.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * float.getPi() * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = { t1, t2, sqrt(max(0, 1 - t1 * t1 - t2 * t2)) };

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    Frame hemi_frame = Frame(hemi_dir_in);
    float3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return factor * normalize(float3(alpha * hemi_N.x, alpha * hemi_N.y, max(0, hemi_N.z)));
}

[ForceInline]
float3 sample_visible_normals(float3 local_dir_in, float alpha_x, float alpha_y, float2 rnd_param)
{
    float factor = 1;
    // The incoming direction is in the "ellipsodial configuration" in Heitz's paper
    if (local_dir_in.z < 0) {
        // Ensure the input is on top of the surface.
        local_dir_in = -local_dir_in;
        factor = -1;
    }

    // Transform the incoming direction to the "hemisphere configuration".
    float3 hemi_dir_in = normalize(float3(alpha_x * local_dir_in.x, alpha_y * local_dir_in.y, local_dir_in.z));

    // Parameterization of the projected area of a hemisphere.
    // First, sample a disk.
    float r = sqrt(rnd_param.x);
    float phi = 2 * float.getPi() * rnd_param.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    // Vertically scale the position of a sample to account for the projection.
    float s = (1 + hemi_dir_in.z) / 2;
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    // Point in the disk space
    float3 disk_N = {t1, t2, sqrt(max(0, 1 - t1*t1 - t2*t2))};

    // Reprojection onto hemisphere -- we get our sampled normal in hemisphere space.
    Frame hemi_frame = Frame(hemi_dir_in);
    float3 hemi_N = to_world(hemi_frame, disk_N);

    // Transforming the normal back to the ellipsoid configuration
    return factor * normalize(float3(alpha_x * hemi_N.x, alpha_y * hemi_N.y, max(0, hemi_N.z)));
}
