from memgpt.agent import Agent

def move(self: Agent, target: str ) -> str:
    """
    Move to the target.

    This function moves the robot to a predefined target.
    After you call this function, ask me if I need help or just want to chat.

    Args:
        target (str): the target to move to, MUST be uppercase.

    Returns:
        str: A string indicating the target the robot is moving to.

    Example:
        >>> move(target='SOFA')
        Moving to sofa.  # This is an example output and the target may vary each time the function is called.
    """
    import subprocess

    # Define the command as a list
    command = ["/usr/local/webots/webots-controller", "/home/disky/webots_projects/apartment/controllers/rosbot/agent_rl.py", "-target", target]

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # output_string = f"Moving to {target}."
    return result


from memgpt import create_client 

client = create_client() 
tool = client.create_tool(move, name='move', update=True, tags=['base']) 