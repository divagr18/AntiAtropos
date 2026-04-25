import re

path = r'C:\Users\kesha\AppData\Local\uv\cache\archive-v0\XqdAvZFy3eRi9W25WFXDP\Lib\site-packages\colab_mcp\session.py'
with open(path, 'r') as f:
    content = f.read()

# The corrupted version from the failed PowerShell replacement
old = '''async def check_session_proxy_tool_fn(random_string: str = " \\, ctx: Context = CurrentContext()) -> bool:
 \\\\\\Opens a connection to a Google Colab browser session.

 Args:
 random_string: A dummy parameter required by some MCP clients for
 tools with no real arguments. This value is ignored.
 \\\\\\
 fe_connected'''

new = '''async def check_session_proxy_tool_fn(random_string: str = "", ctx: Context = CurrentContext()) -> bool:
    """Opens a connection to a Google Colab browser session.

    Args:
        random_string: A dummy parameter required by some MCP clients for
            tools with no real arguments. This value is ignored.
    """
    fe_connected'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Fixed corrupted version successfully.')
else:
    print('Old pattern not found. Checking current state...')
    # Check if the file still has the original version or corrupted version
    if 'async def check_session_proxy_tool_fn(ctx: Context = CurrentContext()) -> bool:' in content:
        print('Found original version, applying fix...')
        old2 = 'async def check_session_proxy_tool_fn(ctx: Context = CurrentContext()) -> bool:\n    fe_connected'
        new2 = 'async def check_session_proxy_tool_fn(random_string: str = "", ctx: Context = CurrentContext()) -> bool:\n    """Opens a connection to a Google Colab browser session.\n\n    Args:\n        random_string: A dummy parameter required by some MCP clients for\n            tools with no real arguments. This value is ignored.\n    """\n    fe_connected'
        content = content.replace(old2, new2)
        with open(path, 'w') as f:
            f.write(content)
        print('Fixed original version successfully.')
    else:
        # Print the current function for debugging
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'check_session_proxy_tool_fn' in line:
                print(f'Line {i}: {line}')
        print('Could not find expected pattern.')
