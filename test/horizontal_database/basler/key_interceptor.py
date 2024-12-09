from pynput import keyboard
import subprocess
import os

# Track currently pressed keys
current_keys = set()

# State management
processing_in_progress = False      # Prevent multiple simultaneous authorizations
pending_action = None               # Store the pending action to be processed after authorization
failed_authorizations = set()       # Track actions with failed authorizations

def simulate_key_press(keys):
    """
    Simulate the authorized shortcut press.
    Used only for reversing actions on authorization failure.
    """
    keys_combination = " ".join(keys)
    print(f"Simulating key press: {keys_combination}")
    os.system(f"xdotool key {keys_combination}")

def reverse_action(action, keys):
    """
    Reverse the action by simulating the necessary key presses.
    """
    print(f"Reversing action for '{action}'")
    if action == "Win":
        # Simulate pressing Escape to close the Start menu
        simulate_key_press(["Super_L"])
    elif action == "Win+Tab":
        # Simulate pressing Win+Tab again to close the task switcher
        simulate_key_press(["Super_L", "Tab"])
    elif action == "Ctrl+Alt+Del":
        # Simulate pressing Escape to cancel the prompt
        simulate_key_press(["Escape"])
    else:
        print(f"No reversal action defined for '{action}'.")

def request_authorization(action, keys):
    """
    Request system authorization.
    Execute the key's action only after successful authorization.
    On failure, reverse the action and mark as failed.
    """
    global processing_in_progress, pending_action


    if processing_in_progress:
        print("Authorization already in progress, ignoring new requests.")
        return False

    processing_in_progress = True
    print(f"Requesting authorization for '{action}'...")
    try:
        # Run pkexec with a timeout to prevent infinite retries
        result = subprocess.run(
            ['pkexec', '/bin/echo', 'Authorized'],  # Replace '/bin/echo' with the actual command if needed
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10  # Timeout in seconds
        )
    except subprocess.TimeoutExpired:
        print(f"Authorization timed out for '{action}'. Treating as failed.")
        reverse_action(action, keys)
        processing_in_progress = False
        pending_action = None
        return False

    processing_in_progress = False

    if result.returncode == 0:
        print(f"Authorization granted for '{action}'.")
        # Do NOT simulate key press since the user already pressed it
        pending_action = None  # Clear pending action

        return True
    else:
        print(f"Authorization denied for '{action}'.")
        reverse_action(action, keys)  # Reverse the action on denial
        pending_action = None  # Clear pending action even if denied
        return False

def detect_combination():
    """
    Detect key combinations and decide the action.
    """
    global pending_action
    # Define all possible combinations
    combinations = {
        frozenset([keyboard.Key.cmd, keyboard.Key.tab]): ("Win+Tab", ["Super_L", "Tab"]),
        frozenset([keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.delete]): ("Ctrl+Alt+Del", ["Control_L", "Alt_L", "Delete"]),
    }
    
    # Check for defined combinations first
    for combo_keys, (action_name, keys_list) in combinations.items():
        if combo_keys.issubset(current_keys):
            print(f"Detected combination: {action_name}")
            pending_action = (action_name, keys_list)
            return

    # Check for single key if no combination matched
    if len(current_keys) == 1 and keyboard.Key.cmd in current_keys:
        print("Detected single key: Win")
        pending_action = ("Win", ["Super_L"])

def process_pending_action():
    """
    Process the pending action if authorization is not already in progress.
    """
    global pending_action
    if pending_action and not processing_in_progress:
        action, keys = pending_action
        print(f"Processing action: {action}")
        authorization_success = request_authorization(action, keys)
        if not authorization_success:
            print(f"Action '{action}' failed authorization and was reversed.")
        else:
            print(f"Action '{action}' authorized successfully.")

def on_press(key):
    """
    Handle key press events.
    """
    global processing_in_progress, failed_authorizations
    if processing_in_progress:
        print(f"Ignoring key press during authorization: {key}")
        return

    # If the user presses the keys again after a failed authorization, allow retry
    # Identify which action is being attempted based on current_keys
    # Clear the specific action from failed_authorizations to allow retry
    # Iterate through combinations to find matching actions
    actions = {
        frozenset([keyboard.Key.cmd, keyboard.Key.tab]): "Win+Tab",
        frozenset([keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.delete]): "Ctrl+Alt+Del",
    }
    single_actions = {
        frozenset([keyboard.Key.cmd]): "Win",
    }

    # Check for combination first
    for combo_keys, action_name in actions.items():
        if combo_keys.issubset(current_keys):
            if action_name in failed_authorizations:
                print(f"User is attempting '{action_name}' again. Clearing failed authorization to allow retry.")
                failed_authorizations.discard(action_name)
            break
    else:
        # Check for single actions
        for single_key, action_name in single_actions.items():
            if single_key.issubset(current_keys):
                if action_name in failed_authorizations:
                    print(f"User is attempting '{action_name}' again. Clearing failed authorization to allow retry.")
                    failed_authorizations.discard(action_name)
                break

    current_keys.add(key)
    print(f"Key pressed: {key}, current_keys: {list(current_keys)}")

    # Detect combinations based on pressed keys
    detect_combination()

def on_release(key):
    """
    Handle key release events.
    """
    global pending_action, processing_in_progress
    if processing_in_progress:
        print(f"Ignoring key release during authorization: {key}")
        return

    current_keys.discard(key)
    print(f"Key released: {key}, current_keys: {list(current_keys)}")

    # Process the pending action after all keys are released
    if pending_action and not current_keys:
        process_pending_action()

def main():
    print("Listening for shortcuts...")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    main()
