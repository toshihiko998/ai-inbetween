def run_pipeline(
    image_a,
    image_b,
    out_dir,
    n_frames,
    thickness=1,
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    img_a = load_grayscale(image_a)
    img_b = load_grayscale(image_b)

    bin_a = preprocess(img_a)
    bin_b = preprocess(img_b)

    strokes_a = extract_strokes(bin_a)
    strokes_b = extract_strokes(bin_b)

    matches = match_strokes(strokes_a, strokes_b)
    if not matches:
        print("No matches found.")
        return

    # ---- 主線 + 補助線 選別 ----
    matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
    main_matches = matches[:1]      # 主線1本
    sub_matches = matches[1:4]      # 補助線 最大3本

    for i in range(1, n_frames + 1):
        alpha = i / (n_frames + 1)

        polylines = []

        # 主線
        for m in main_matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points
            pa = resample_polyline(sa, 64)
            pb = resample_polyline(sb, 64)
            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append((interp, 1))

        # 補助線（薄く）
        for m in sub_matches:
            sa = strokes_a[m.a_index].points
            sb = strokes_b[m.b_index].points
            pa = resample_polyline(sa, 64)
            pb = resample_polyline(sb, 64)
            interp = (1.0 - alpha) * pa + alpha * pb
            polylines.append((interp, 1))

        frame = render_polylines(polylines, img_a.shape, thickness=1)

        out_path = os.path.join(out_dir, f"{i:04d}.png")
        cv2.imwrite(out_path, frame)

        print(f"[frame {i:04d}] polylines={len(polylines)}")
