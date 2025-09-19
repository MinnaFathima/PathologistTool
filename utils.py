def stitch_heatmap(coords, heatmaps, image_size, tile_size=224):
    # coords: list of (x1,y1,x2,y2), heatmaps list of 2D arrays tile_size x tile_size
    H, W = image_size[1], image_size[0]
    agg = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    for (x1,y1,x2,y2), hmap in zip(coords, heatmaps):
        h = hmap.shape[0]
        w = hmap.shape[1]
        # resize heatmap to box size in case padding used
        hmap_resized = cv2.resize(hmap, (x2-x1, y2-y1))
        agg[y1:y2, x1:x2] += hmap_resized
        weight[y1:y2, x1:x2] += 1.0
    weight[weight==0] = 1.0
    stitched = agg / weight
    stitched = (stitched - stitched.min())
    if stitched.max()!=0:
        stitched = stitched / stitched.max()
    return stitched


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5):
    img = np.array(pil_img).astype(np.uint8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (heatmap_color * 0.7 + img * 0.3).astype(np.uint8)
    blended = ((1-alpha) * img + alpha * heatmap_color).astype(np.uint8)
    return Image.fromarray(blended)


def pil_to_tensor(image_pil, transform):
    return transform(image_pil).unsqueeze(0)