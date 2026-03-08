
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
