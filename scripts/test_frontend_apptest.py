from streamlit.testing.v1 import AppTest
import os

def main():
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'app.py'))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    print("Initial App run successfully!")
    print("Current Page:", at.session_state['current_page'])
    print("Markdown count:", len(at.markdown))

    # 1. Verify Landing Page content
    found_hero = any('AI-Assisted Bone Fracture Analysis' in m.value for m in at.markdown)
    found_caps = any('Clinical Capabilities' in m.value for m in at.markdown)
    found_how = any('Diagnostic Evaluation Workflow' in m.value for m in at.markdown)
    found_disclaimer = any('Medical Disclaimer' in m.value for m in at.markdown)
    print(f"Landing Page Elements -> Hero: {found_hero}, Capabilities: {found_caps}, Workflow: {found_how}, Disclaimer: {found_disclaimer}")
    assert found_hero and found_caps and found_how and found_disclaimer, "Landing page elements missing!"

    # 2. Test Navigation to Analysis Workspace
    btn_analysis = [b for b in at.button if b.label == 'Analysis Workspace'][0]
    btn_analysis.click().run()
    print("\nNavigated to Analysis Page!")
    print("Current Page:", at.session_state['current_page'])

    # 3. Test clicking 'Wrist X-ray (AP)' preset
    btn_wrist = [b for b in at.button if b.label == 'Wrist X-ray (AP)'][0]
    btn_wrist.click().run()
    print("\nClicked Wrist Preset!")
    print("Uploaded bytes present:", at.session_state['uploaded_bytes'] is not None)
    print("Uploaded filename:", at.session_state['uploaded_filename'])

    # 4. Test running Diagnostic Analysis
    btn_run = [b for b in at.button if b.label == 'Analyze Radiograph'][0]
    btn_run.click().run()
    print("\nExecuted Diagnostic Analysis!")
    print("Prediction result present:", at.session_state['prediction_result'] is not None)
    res = at.session_state['prediction_result']
    print("Predicted Region:", res['anatomical_region'])
    print("Fracture Detected:", res['fracture'])
    print("Fracture Confidence:", res['fracture_confidence'])
    print("Caption:", res['caption'])

    # 5. Verify Clean Diagnostic UI & Hierarchy
    analysis_text = ' '.join([m.value for m in at.markdown])
    
    # Check that required clean cards exist
    has_region_card = "Anatomical Region" in analysis_text
    has_frac_card = "Fracture Assessment" in analysis_text
    has_summary_card = "Factual Summary" in analysis_text
    has_disclaimer = "Research Disclaimer" in analysis_text
    print("\nDiagnostic Hierarchy Checks:")
    print(f"  Anatomical Region Card: {has_region_card}")
    print(f"  Fracture Assessment Card: {has_frac_card}")
    print(f"  Factual Summary Card: {has_summary_card}")
    print(f"  Research Disclaimer: {has_disclaimer}")
    assert has_region_card and has_frac_card and has_summary_card, "Missing diagnostic cards!"

    # 6. Verify that debug/unwanted details are REMOVED
    has_cand_regions = "Inspect Alternative Candidate Regions" in analysis_text
    has_thresh_text = "Calibrated decision threshold" in analysis_text
    has_calc_frac_prob = "Calculated Fracture Probability" in analysis_text
    print("\nDebug Removal Checks:")
    print(f"  'Inspect Alternative Candidate Regions' present: {has_cand_regions}")
    print(f"  'Calibrated decision threshold' present: {has_thresh_text}")
    print(f"  'Calculated Fracture Probability' present: {has_calc_frac_prob}")
    assert not has_cand_regions, "'Inspect Alternative Candidate Regions' was NOT removed!"
    assert not has_thresh_text, "'Calibrated decision threshold' was NOT removed!"
    assert not has_calc_frac_prob, "'Calculated Fracture Probability' was NOT removed!"

    # 7. Verify absence of localization UI
    print("\nLocalization UI Checks:")
    print("  'Localization Status' in UI:", 'Localization Status' in analysis_text)
    print("  'Fracture Spatial Localization' in UI:", 'Fracture Spatial Localization' in analysis_text)
    print("  'bounding box' in UI:", 'bounding box' in analysis_text.lower())
    print("  'lesion' in UI:", 'lesion' in analysis_text.lower())
    assert 'Localization Status' not in analysis_text
    assert 'Fracture Spatial Localization' not in analysis_text
    assert 'bounding box' not in analysis_text.lower()
    assert 'lesion' not in analysis_text.lower()

    # 8. Verify No 4-space indented lines inside HTML markdown blocks
    # (which would trigger marked.js indented code blocks <pre><code>)
    for idx, m in enumerate(at.markdown):
        val = m.value
        # Skip the root CSS style block
        if "<style>" in val:
            continue
        lines = val.splitlines()
        for line_no, l in enumerate(lines):
            assert not l.startswith("    "), f"Found 4-space indented line in markdown #{idx} line #{line_no}: {repr(l)}"
            assert not l.startswith("\t"), f"Found tab indented line in markdown #{idx} line #{line_no}: {repr(l)}"
    print("Indented Code Block Prevention Check: 100% Passed (Zero 4-space/tab indented lines)")

    # 9. Test Clear Image button
    btn_clear = [b for b in at.button if b.label == 'Clear Image'][0]
    btn_clear.click().run()
    print("\nClicked Clear Image Button!")
    print("Uploaded bytes after clear:", at.session_state['uploaded_bytes'])
    print("Prediction result after clear:", at.session_state['prediction_result'])
    assert at.session_state['uploaded_bytes'] is None, "Uploaded bytes not cleared!"
    assert at.session_state['prediction_result'] is None, "Prediction result not cleared!"

    # 10. Test Hand X-ray (PA) preset
    btn_hand = [b for b in at.button if b.label == 'Hand X-ray (PA)'][0]
    btn_hand.click().run()
    print("\nClicked Hand X-ray Preset!")
    btn_run = [b for b in at.button if b.label == 'Analyze Radiograph'][0]
    btn_run.click().run()
    res_hand = at.session_state['prediction_result']
    print("Hand Result -> Region:", res_hand['anatomical_region'], "| Fracture:", res_hand['fracture'], "| Prob:", f"{res_hand['fracture_confidence']*100:.1f}%")
    assert res_hand['anatomical_region'] == 'Hand', f"Expected Hand, got {res_hand['anatomical_region']}"
    assert not res_hand['fracture'], "Expected normal hand (no fracture)!"

    # Verify no raw HTML bug on normal hand result
    hand_text = ' '.join([m.value for m in at.markdown])
    assert "NO FRACTURE DETECTED" in hand_text, "NO FRACTURE DETECTED badge missing!"

    # 11. Test Navigation to About Page
    btn_about = [b for b in at.button if b.label == 'About System'][0]
    btn_about.click().run()
    print("\nNavigated to About Page!")
    print("Current Page:", at.session_state['current_page'])
    found_about = any('Technical Specifications' in m.value for m in at.markdown)
    print(f"About Page Rendered: {found_about}")
    assert found_about, "About page did not render!"

    # 12. Return to Landing
    btn_home = [b for b in at.button if b.label == 'Home'][0]
    btn_home.click().run()
    print("\nReturned to Landing Page!")
    print("Current Page:", at.session_state['current_page'])
    print("\nALL VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
