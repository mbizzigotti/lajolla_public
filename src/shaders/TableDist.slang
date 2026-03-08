import shared;
import Scene;

extension FloatArray
{
	__subscript(int index) -> float
    {
        get { return scene.values[offset + index]; }
	}
}

int sample(in TableDist1D table, float rnd_param)
{
    int size = table.pmf.count;
    // float *ptr = std::upper_bound(table.cdf.data(), table.cdf.data() + size + 1, rnd_param);
    // int offset = std::clamp(int(ptr - table.cdf.data() - 1), 0, size - 1);
    // return offset;
    return 0;
}

float pmf(in TableDist1D table, int id)
{
    return table.pmf[id];
}

float2 sample(in TableDist2D table, float2 rnd_param)
{
    int w = table.width, h = table.height;
    // We first sample a row from the marginal distribution
   // const Real *y_ptr = std::upper_bound(
   //     table.cdf_marginals.data(),
   //     table.cdf_marginals.data() + h + 1,
   //     rnd_param[1]);
   // int y_offset = std::clamp(int(y_ptr - table.cdf_marginals.data() - 1), 0, h - 1);
   // // Uniformly remap rnd_param[1] to find the continuous offset 
   // Real dy = rnd_param[1] - table.cdf_marginals[y_offset];
   // if ((table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]) > 0) {
   //     dy /= (table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]);
   // }
   // // Sample a column at the row y_offset
   // const Real *cdf = &table.cdf_rows[y_offset * (w + 1)];
   // const Real *x_ptr = std::upper_bound(cdf, cdf + w + 1, rnd_param[0]);
   // int x_offset = std::clamp(int(x_ptr - cdf - 1), 0, w - 1);
   // // Uniformly remap rnd_param[0]
   // Real dx = rnd_param[0] - cdf[x_offset];
   // if (cdf[x_offset + 1] - cdf[x_offset] > 0) {
   //     dx /= (cdf[x_offset + 1] - cdf[x_offset]);
   // }
   // return Vector2 { (x_offset + dx) / w, (y_offset + dy) / h };
    return float2(0);
}

float pdf(in TableDist2D table, float2 xy) {
    // Convert xy to integer rows & columns
    int w = table.width, h = table.height;
    int x = int(clamp(xy.x * float(w), 0, float(w - 1)));
    int y = int(clamp(xy.y * float(h), 0, float(h - 1)));
    // What's the PDF for sampling row y?
    float pdf_y = table.pdf_marginals[y];
    // What's the PDF for sampling row x?
    float pdf_x = table.pdf_rows[y * w + x];
    return pdf_y * pdf_x * w * h;
}
