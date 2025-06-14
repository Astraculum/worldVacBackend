# WorldVac Backend API Documentation

## Base URL

The base URL for all API endpoints is configurable. By default, it runs on `http://localhost:8000`.

## Authentication

Most endpoints require authentication using a Bearer token. Include the token in the Authorization header:

```plain
Authorization: Bearer <your_token>
```

The token is obtained through the `/auth/register` or `/auth/login` endpoints and is valid for 30 days.

## Endpoints

### 1. Authentication

#### Register User

- **URL**: `/auth/register`
- **Method**: `POST`
- **Description**: Register a new user and get an access token
- **Request Body**:

```json
{
  "username": "string",
  "password": "string"
}
```

- **Response**: `RegisterResponse`

```json
{
  "success": true,
  "message": "注册成功",
  "token": "string",
  "id": "string",
  "user": {
    "id": "string",
    "username": "string"
  }
}
```

- **Error Responses**:
  - `400 Bad Request`: If username already exists
  - `500 Internal Server Error`: If user creation fails

#### Login

- **URL**: `/auth/login`
- **Method**: `POST`
- **Description**: Login and get an access token
- **Request Body**:

```json
{
  "username": "string",
  "password": "string"
}
```

- **Response**: `LoginResponse`

```json
{
  "success": true,
  "message": "登录成功",
  "token": "string",
  "id": "string",
  "user": {
    "id": "string",
    "username": "string"
  }
}
```

- **Error Responses**:
  - `400 Bad Request`: If username doesn't exist
  - `401 Unauthorized`: If password is incorrect or token is invalid

### 2. User Management

#### Get User Home

- **URL**: `/user/{user_id}`
- **Method**: `GET`
- **Description**: Get user's information and their worlds
- **Response**:

```json
{
  "user_id": "string",
  "username": "string",
  "worlds": [
    {
      "world_id": "string",
      "commit_id": "string"
    }
  ]
}
```

- **Error Responses**:
  - `404 Not Found`: If user doesn't exist

### 3. World Management

#### Get World Home

- **URL**: `/user/{user_id}/world/{world_id}`
- **Method**: `GET`
- **Description**: 获取世界信息及其所有提交记录 (Get world information and its commits)
- **Response**:

```json
{
  "user_id": "string",      // 用户ID
  "world_id": "string",     // 世界ID
  "commits": [              // 提交记录列表
    {
      "commit_id": "string",        // 提交ID
      "event_summary": "string",    // 事件摘要
      "topic": "string"            // 主题
    }
  ],
  "latest_commit": {        // 最新的提交记录
    "commit_id": "string",         // 提交ID
    "event_summary": "string",     // 事件摘要
    "topic": "string"             // 主题
  }
}
```

- **Error Responses**:
  - `404 Not Found`: 如果世界不存在 (If world doesn't exist)

#### Get World State

- **URL**: `/user/{user_id}/world/{world_id}/commit/{commit_id}`
- **Method**: `GET`
- **Description**: Get detailed world state for a specific commit. This endpoint handles both world initialization and scene initialization states.
- **Response**: `WorldModel` or initialization status

```json
// When world is initializing
{
    "user_id": "string",
    "world_id": "string",
    "commit_id": "string",
    "status": "initializing_world"
}

// When scene is initializing
{
    "user_id": "string",
    "world_id": "string",
    "commit_id": "string",
    "status": "initializing_scene"
}

// When waiting for dialogues
{
    "status": "waiting_for_dialogues"
}

// When waiting for options
{
    "status": "waiting_for_options"
}

// When world is ready
{
    "id": "string",
    "commit_id": "string",
    "title": "string",
    "crisis": "string",
    "allCharacters": [
        {
            "id": "string",
            "name": "string",
            "description": "string"
        }
    ],
    "currentScene": {
        "heading": "string",
        "location": "string",
        "characterIds": ["string"],
        "eventList": [],
        "missionList": []
    },
    "worldNews": [
        {
            "id": "string",
            "title": "string",
            "content": "string",
            "timestamp": "string",
            "impact": "string",
            "category": "string",
            "relatedLocation": "string"
        }
    ],
    "worldCharacteristics": [
        {
            "name": "string",
            "description": "string"
        }
    ]
}
```

- **Error Responses**:
  - `404 Not Found`: If the world or commit doesn't exist
  - `500 Internal Server Error`: If world or scene initialization fails

#### Create World from Seed Prompt

- **URL**: `/world/seed_prompt_to_world`
- **Method**: `POST`
- **Description**: Create a new world from a seed prompt
- **Request Body**: `SeedPromptToWorldModel`

```json
{
  "user_id": "string",
  "seed_prompt": "string"
}
```

- **Response**:

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `500 Internal Server Error`: If world creation fails

#### Create World

- **URL**: `/world/create_world`
- **Method**: `POST`
- **Description**: Create a new world with custom parameters
- **Request Body**: `CreateWorldModel`

```json
{
  "user_id": "string",
  "protagonist_description": "string",
  "strategy": "string",
  "world_state": "string",
  "geography_info": "string",
  "tone": "string",
  "max_rounds": 20
}
```

- **Response**:

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Error Responses**:
  - `403 Forbidden`: If the user_id doesn't match the authenticated user
  - `500 Internal Server Error`: If world creation fails

#### Get All Worlds

- **URL**: `/world/get_all_worlds`
- **Method**: `POST`
- **Description**: Get all world identifiers for the current user
- **Request Body**: `GetAllWorldsModel`

```json
{
  "user_id": "string"
}
```

- **Response**: `list[WorldIdentifier]`
- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user

#### Delete World Commit

- **URL**: `/world/delete_world_commit`
- **Method**: `POST`
- **Description**: Delete a specific world commit
- **Request Body**: `DeleteWorldCommitModel`

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Response**:

```json
{
  "message": "World commit deleted successfully"
}
```

- **Error Responses**:
  - `403 Forbidden`: If the user_id doesn't match the authenticated user
  - `404 Not Found`: If the world commit doesn't exist

#### Make World Public

- **URL**: `/world/public_world`
- **Method**: `POST`
- **Description**: Make a world public so other users can fork it
- **Request Body**: `PublicWorldModel`

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Response**:

```json
{
  "user_id": "string",
  "world_id": "string",
  "status": "publishing_world"
}
```

- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world commit doesn't exist

#### Fork Public World

- **URL**: `/world/fork`
- **Method**: `POST`
- **Description**: Fork a public world to create your own copy
- **Request Body**: `ForkWorldModel`

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string",
  "fork_seed_prompt": "string", 
  "mode": "full"          // or "remodify"       
}
```

- **Response**:

```json
{
  "user_id": "string",
  "world_id": "string",
  "status": "forking_world"
}
```

- **Error Responses**:
  - `403 Forbidden`: If trying to fork a non-public world
  - `404 Not Found`: If the world commit doesn't exist

#### Get All Public Worlds

- **URL**: `/world/public_worlds`
- **Method**: `GET`
- **Description**: Get all public worlds available for forking
- **Response**: List of public world identifiers

```json
[
  {
    "user_id": "string",
    "world_id": "string",
    "commit_id": "string"
  }
]
```

- **Error Responses**:
  - `401 Unauthorized`: If not authenticated

### 4. Scene Management

#### Get Scene Events

- **URL**: `/{user_id}/{world_id}/{commit_id}/scene/events`
- **Method**: `POST`
- **Description**: Get events in the current scene. This endpoint returns the current scene state including dialogues, options, participants, and missions.
- **Response**: `SceneModel` or error status

```json
// When scene is ready
{
  "heading": "string",
  "location": "string",
  "characterIds": ["string"],
  "eventList": [
    {
      "type": "string",
      "content": "string",
      "characterId": "string",
      "timestamp": "string"
    }
  ],
  "missionList": [
    {
      "id": "string",
      "name": "string",
      "description": "string",
      "status": "string",
      "mission_type": "string"
    }
  ]
}
```

- **Error Responses**:
  - `400 Bad Request`: If scene is not started, terminated, or in an unknown state
  - `400 Bad Request`: If dialogues, options, or participants are not generated
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

#### Check Event Generation Status

- **URL**: `/{user_id}/{world_id}/{commit_id}/scene/is-event-generated`
- **Method**: `POST`
- **Description**: Check if all events (dialogues, options, participants) have been generated for the current scene
- **Response**: `boolean`
- **Error Responses**:
  - `400 Bad Request`: If scene is terminated, not started, or in an unknown state
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

#### Select Option

- **URL**: `/{user_id}/{world_id}/{commit_id}/scene/select-option`
- **Method**: `POST`
- **Description**: Select an option in the current scene
- **Request Body**: `SelectOptionModel`

```json
{
  "option_index": 0
}
```

- **Response**:

```json
{
  "status": "success"
}
```

- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

#### Input Action

- **URL**: `/{user_id}/{world_id}/{commit_id}/scene/input-action`
- **Method**: `POST`
- **Description**: Input an action in the current scene
- **Request Body**: `InputActionModel`

```json
{
  "action": "string"
}
```

- **Response**:

```json
{
  "status": "success"
}
```

- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

#### Check Scene Status

- **URL**: `/{user_id}/{world_id}/{commit_id}/scene/is_finished`
- **Method**: `POST`
- **Description**: Check if the current scene is finished. If the scene is finished, a new commit ID will be automatically generated and the scene status will be cleared.
- **Request Body**: `IsSceneFinishedModel`

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Response**:

```json
{
    "is_finished": boolean
}
```

- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

### 5. Character Management

#### Get Characters

- **URL**: `/{user_id}/{world_id}/{commit_id}/world/get_characters`
- **Method**: `GET`
- **Description**: Get specific characters by their IDs
- **Request Body**: `GetCharactersModel`

```json
{
  "ids": ["string"]
}
```

- **Response**: `list[CharacterModel]`
- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

#### Get All Characters

- **URL**: `/{user_id}/{world_id}/{commit_id}/world/get_all_characters`
- **Method**: `GET`
- **Description**: Get all characters in the world
- **Request Body**: `GetAllCharactersModel`

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Response**: `list[CharacterModel]`
- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

#### Get Player Character

- **URL**: `/{user_id}/{world_id}/{commit_id}/world/get_player_character`
- **Method**: `GET`
- **Description**: Get the player's character
- **Request Body**: `GetPlayerCharacterModel`

```json
{
  "user_id": "string",
  "world_id": "string",
  "commit_id": "string"
}
```

- **Response**: `CharacterModel`
- **Error Responses**:
  - `403 Forbidden`: If user_id doesn't match the authenticated user
  - `404 Not Found`: If the world or commit doesn't exist

## Error Responses

The API uses standard HTTP status codes:

- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

### Common Error Messages

- `400 Bad Request`: 
  - "用户已存在" (User already exists)
  - "用户不存在" (User does not exist)
  - "密码错误" (Incorrect password)
  - "场景未开始" (Scene not started)
  - "场景已结束" (Scene has ended)
  - "场景状态未知" (Unknown scene status)
  - "对话未生成" (Dialogues not generated)
  - "选项未生成" (Options not generated)
  - "角色未生成" (Characters not generated)

- `401 Unauthorized`:
  - "登录失败: {error}" (Login failed: {error})

- `403 Forbidden`:
  - "user_id不匹配" (User ID mismatch)
  - "user_id不匹配 无权限删除" (User ID mismatch, no permission to delete)
  - "无权限访问该世界" (No permission to access this world)

- `404 Not Found`:
  - "用户不存在" (User does not exist)
  - "世界不存在" (World does not exist)
  - "World not found"
  - "Commit not found"
  - "World commit not found"

- `500 Internal Server Error`:
  - "生成世界失败: {error}" (World generation failed: {error})
  - "World initialization failed: {error}"
  - "Scene initialization failed: {error}"
  - "Commit creation failed: {error}"

## Additional Notes

### World Visibility

Worlds can have different visibility levels:
- `PRIVATE`: Only accessible by the owner
- `PUBLIC`: Accessible by all users
- `SHARED`: Accessible by the owner and specifically shared users

### Task Status

Several operations (world creation, scene initialization, commit creation, forking) are asynchronous and return a status indicating the operation is in progress. The client should poll the appropriate endpoints to check the completion status.

### File Storage

The backend stores world data in two main directories:
- `worlds-json/`: Contains world state files
- `commit-trees-json/`: Contains commit tree information

### Authentication

The authentication system uses JWT tokens that are valid for 30 days. If a token expires, the system will automatically generate a new token during login.

### CORS Support

The API supports CORS with the following configuration:
- All origins are allowed (`*`)
- Credentials are allowed
- All standard HTTP methods are supported
- All headers are allowed

### SSL Support

The server can be configured to run with SSL/TLS by providing:
- SSL key file path
- SSL certificate file path

If SSL is not configured, the server will run on HTTP with a warning message.
