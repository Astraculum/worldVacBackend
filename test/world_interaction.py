#!/usr/bin/env python3
import json
import subprocess
import time
import argparse
import random

def run_curl_command(command):
    """Execute a curl command and return the response"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing command: {result.stderr}")
            return None
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error parsing JSON response: {result.stdout}") # type: ignore
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def login(host, port, username, password):
    """Login and get authentication token"""
    login_cmd = f'''curl -X POST "http://{host}:{port}/auth/login" \\
        -H "Content-Type: application/json" \\
        -d '{{"username": "{username}", "password": "{password}"}}'
    '''
    response = run_curl_command(login_cmd)
    if not response or not response.get('token'):
        print("Failed to login")
        return None
    return response['token'], response['id']

def get_world_info(host, port, token, user_id, world_id, commit_id):
    """Get information about the specified world"""
    world_cmd = f'''curl -X GET "http://{host}:{port}/user/{user_id}/world/{world_id}/commit/{commit_id}" \\
        -H "Authorization: Bearer {token}"
    '''
    return run_curl_command(world_cmd)

def get_scene_events(host, port, token, user_id, world_id, commit_id):
    """Get current scene events"""
    events_cmd = f'''curl -X POST "http://{host}:{port}/{user_id}/{world_id}/{commit_id}/scene/events" \\
        -H "Authorization: Bearer {token}"
    '''
    return run_curl_command(events_cmd)

def select_option(host, port, token, user_id, world_id, commit_id, option_index):
    """Select an option in the current scene"""
    select_cmd = f'''curl -X POST "http://{host}:{port}/{user_id}/{world_id}/{commit_id}/scene/select-option" \\
        -H "Authorization: Bearer {token}" \\
        -H "Content-Type: application/json" \\
        -d '{{"option_index": {option_index}}}'
    '''
    return run_curl_command(select_cmd)

def check_scene_finished(host, port, token, user_id, world_id, commit_id):
    """Check if the current scene is finished"""
    check_cmd = f'''curl -X POST "http://{host}:{port}/{user_id}/{world_id}/{commit_id}/scene/is_finished" \\
        -H "Authorization: Bearer {token}"
    '''
    return run_curl_command(check_cmd)

def check_event_generated(host, port, token, user_id, world_id, commit_id):
    """Check if events have been generated for the current scene"""
    check_cmd = f'''curl -X POST "http://{host}:{port}/{user_id}/{world_id}/{commit_id}/scene/is-event-generated" \\
        -H "Authorization: Bearer {token}"
    '''
    return run_curl_command(check_cmd)

def main():
    parser = argparse.ArgumentParser(description='Interact with the world using curl commands')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', default='8000', help='Server port')
    parser.add_argument('--username', required=True, help='Username')
    parser.add_argument('--password', required=True, help='Password')
    parser.add_argument('--world-id', required=True, help='World ID')
    parser.add_argument('--commit-id', required=True, help='Commit ID')
    parser.add_argument('--option-index', type=int, default=-1, help='Option index to select')
    
    args = parser.parse_args()

    # Login
    print("Logging in...")
    auth_result = login(args.host, args.port, args.username, args.password)
    if not auth_result:
        return
    token, user_id = auth_result
    print(f"Login successful. User ID: {user_id}")

    # Get world info
    print("\nGetting world information...")
    world_info = get_world_info(args.host, args.port, token, user_id, args.world_id, args.commit_id)
    if not world_info:
        print("Failed to get world information")
        return
    print(f"World title: {world_info.get('title', 'Unknown')}")

    while True:
        # Check if scene is finished
        scene_status = check_scene_finished(args.host, args.port, token, user_id, args.world_id, args.commit_id)
        if scene_status and scene_status.get('is_finished', False):
            print("\nScene finished!")
            if scene_status.get('commit_id'):
                print(f"New commit ID: {scene_status['commit_id']}")
            elif scene_status.get('status') == 'creating_new_commit':
                print("Creating new commit...")
            break

        # Get scene events
        print("\nGetting scene events...")
        max_retries = 1000
        retry_count = 0
        while retry_count < max_retries:
            events = get_scene_events(args.host, args.port, token, user_id, args.world_id, args.commit_id)
            if not events:
                print("Failed to get scene events")
                return
            
            if isinstance(events, dict) and events.get('status') == 'initializing_scene':
                print("Scene is initializing, waiting...")
                time.sleep(5)
                continue
            
            # Wait for events to be generated
            is_generated = check_event_generated(args.host, args.port, token, user_id, args.world_id, args.commit_id)
            if not is_generated:
                print("Waiting for events to be generated...")
                time.sleep(2)
                continue

            # Print available options
            if events.get('eventList'):
                print("eventList:")
                for event in events['eventList']:
                    print(event)
                break
            
            retry_count += 1
            time.sleep(2)
        
        if retry_count >= max_retries:
            print("Timeout waiting for scene initialization")
            return

        # Select option
        selected_option_index = args.option_index
        if selected_option_index<0:
            selected_option_index  = random.randint(0,2)
        print(f"\nSelecting option {selected_option_index}...")
        result = select_option(args.host, args.port, token, user_id, args.world_id, args.commit_id, selected_option_index)
        if result and result.get('status') == 'success':
            print("Option selected successfully!")
        else:
            print("Failed to select option")
        time.sleep(2)

if __name__ == "__main__":
    main() 
