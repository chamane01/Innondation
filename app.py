streamlit.errors.StreamlitDuplicateElementKey: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

Traceback:
File "/mount/src/innondation/app.py", line 746, in <module>
    main()
File "/mount/src/innondation/app.py", line 569, in main
    if st.button("Rafraîchir la liste", key="refresh_list"):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/metrics_util.py", line 409, in wrapped_func
    result = non_optional_func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/widgets/button.py", line 238, in button
    return self.dg._button(
           ^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/widgets/button.py", line 907, in _button
    element_id = compute_and_register_element_id(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/lib/utils.py", line 226, in compute_and_register_element_id
    _register_element_id(ctx, element_type, element_id)
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/elements/lib/utils.py", line 127, in _register_element_id
    raise StreamlitDuplicateElementKey(user_key)
