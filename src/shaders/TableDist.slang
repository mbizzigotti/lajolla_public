import Shared;
import Scene;
import Utils;

struct FloatArray
{
    uint offset;
    uint count;

    __subscript(int index) -> float
    {
        get { return scene.values.LoadByteOffset<float>((offset + index) * sizeof(float)); }
    }
}

extension TableDist1D
{
    property FloatArray pmf { get { return { _pmf_cdf[0], _pmf_cdf[1] }; } }
    property FloatArray cdf { get { return { _pmf_cdf[2], _pmf_cdf[3] }; } }
}

extension TableDist2D
{
	property TableDist1D rows      { get { return { _pdf_cdf_rows }; } }
    property TableDist1D marginals { get { return { _pdf_cdf_marginals }; } }
}

int sample(in TableDist1D table, float rnd_param)
{
    int size = table.pmf.count;
    int k = upper_bound(scene.values, table.cdf.offset, table.cdf.count, rnd_param);
    return clamp(int(k - table.cdf.offset - 1), 0, size - 1);
}

float pmf(in TableDist1D table, int id)
{
    return table.pmf[id];
}

float2 sample(in TableDist2D table, float2 rnd_param)
{
    int w = table.width, h = table.height;
    // We first sample a row from the marginal distribution
    int y_index = upper_bound(scene.values, table.marginals.cdf.offset, h + 1, rnd_param[1]);
    int y_offset = clamp(int(y_index - table.marginals.cdf.offset - 1), 0, h - 1);
    // Uniformly remap rnd_param[1] to find the continuous offset 
    float dy = rnd_param[1] - table.marginals.cdf[y_offset];
    if ((table.marginals.cdf[y_offset + 1] - table.marginals.cdf[y_offset]) > 0) {
        dy /= (table.marginals.cdf[y_offset + 1] - table.marginals.cdf[y_offset]);
    }
    // Sample a column at the row y_offset
    int cdf_start = table.rows.cdf.offset + y_offset * (w + 1);
    int x_index = upper_bound(scene.values, cdf_start, w + 1, rnd_param[0]);
    int x_offset = clamp(int(x_index - cdf_start - 1), 0, w - 1);
    // Uniformly remap rnd_param[0]
    float dx = rnd_param[0] - table.rows.cdf[cdf_start + x_offset];
    if (table.rows.cdf[cdf_start + x_offset + 1] - table.rows.cdf[cdf_start + x_offset] > 0) {
        dx /= (table.rows.cdf[cdf_start + x_offset + 1] - table.rows.cdf[cdf_start + x_offset]);
    }
    return { (x_offset + dx) / w, (y_offset + dy) / h };
}

float pdf(in TableDist2D table, float2 xy) {
    // Convert xy to integer rows & columns
    int w = table.width, h = table.height;
    int x = int(clamp(xy.x * float(w), 0, float(w - 1)));
    int y = int(clamp(xy.y * float(h), 0, float(h - 1)));
    // What's the PDF for sampling row y?
    float pdf_y = table.marginals.pmf[y];
    // What's the PDF for sampling row x?
    float pdf_x = table.rows.pmf[y * w + x];
    return pdf_y * pdf_x * w * h;
}
