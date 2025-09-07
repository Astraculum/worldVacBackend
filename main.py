import argparse
import asyncio
import json
import os
import time
import traceback
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask
from starlette.requests import Request

from AgentMatrix.model import (CharacterModel, CommitIdentifier,
                               CreateWorldModel, DeleteWorldCommitModel,
                               ForkRelationModel, ForkWorldModel,
                               GetAllWorldsModel, GetCharactersModel,
                               InputActionModel, LoginModel, LoginResponse,
                               MissionModel, PublicWorldModel, RegisterModel,
                               RegisterResponse, SceneModel,
                               SeedPromptToWorldModel, SelectOptionModel, User,
                               WorldCharacteristicModel, WorldIdentifier,
                               WorldModel, WorldNewsModel, WorldVisibility,
                               character_info_to_model, create_access_token,
                               get_current_user, hash_password,
                               message_to_event_model, verify_password)
from AgentMatrix.src.graph import (ForkRelationEntity, Graph, GroupChatStatus,
                                   HostLayer)
from AgentMatrix.src.llm import LanguageType, LLMClient, LLMConfig, LLMProvider
from AgentMatrix.src.memory import SentenceEmbedding
from AgentMatrix.src.spritesheet_generator import AnnotationParams
from AgentMatrix.src.spritesheet_generator.auto_download import \
    CharacterImageDownloader
from AgentMatrix.src.world import seed_prompt_to_universe_metadata
from backend.utils import start_scene_from_graph
from backend.utils.commit_task import commit_task_manager
from backend.utils.commit_tree import CommitTree
from backend.utils.fork_task import fork_task_manager
from backend.utils.fork_world import background_fork_world
from backend.utils.scene_task import scene_task_manager
from backend.utils.world_task import world_task_manager
from logger import get_logger as get_logger_backend
from logger import set_logger_file as set_logger_file_backend
from logger import set_logger_level as set_logger_level_backend
from AgentMatrix.database.postgre_sql import db
from AgentMatrix.database.sql_base import SQLBaseDB

# 设置logger级别
set_logger_level_backend("DEBUG")

# 设置logger输出到文件
set_logger_file_backend("backendLog.txt", mode="w")

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # 添加PATCH
    allow_headers=[
        "*",
        "Authorization",  # 明确允许Authorization头部
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=["*"],  # 暴露所有响应头部
    max_age=86400,  # 预检请求缓存时间
)

# LLM 配置
GLOBAL_LLM_CONFIG = LLMConfig(
    api_key="NULL",
    model="NULL",
    provider=LLMProvider.SiliconFlow,
    language_type=LanguageType.NotSpecified,
)
GLOBAL_FAST_CHAT_LLM_CONFIG = LLMConfig(
    api_key="NULL",
    model="NULL",
    provider=LLMProvider.SiliconFlow,
    language_type=LanguageType.NotSpecified,
)

CHARACTER_IMAGES_PATH = "character-images"
os.makedirs(CHARACTER_IMAGES_PATH, exist_ok=True)
CHOOSER_TO_AVAILABLE_OPTIONS_PATH = (
    "backend/spritesheet_generator/chooser_to_available_options.json"
)
DICT_CHOOSER_PARAMS_PATH = "backend/spritesheet_generator/chooser_params.json"
GLOBAL_EMBEDDINGS = SentenceEmbedding()
GLOBAL_ANNOTATION_PARAMS = AnnotationParams(
    chooser_to_available_options=json.load(
        open(CHOOSER_TO_AVAILABLE_OPTIONS_PATH, "r")
    ),
    dict_chooser_params=json.load(open(DICT_CHOOSER_PARAMS_PATH, "r")),
)
GLOBAL_CHARACTER_IMAGE_DOWNLOADER = CharacterImageDownloader()
GLOBAL_CHARACTER_IMAGE_DOWNLOADER.start_character_generation_server()




# uuid -> graph

world_dict: dict[WorldIdentifier, Graph] = {}  # (user_id, world_id, commit_id) -> graph
world_lock = asyncio.Lock()
user_dict: dict[str, User] = {}  # user_id -> User
user_lock = asyncio.Lock()
commit_trees_dict: dict[CommitIdentifier, CommitTree] = {}
commit_tree_lock = asyncio.Lock()




async def load_commit_trees():
    """从数据库加载提交树"""
    async with commit_tree_lock:
        trees = await db.get_all_commit_trees()
        for tree in trees:
            commit_trees_dict[
                CommitIdentifier(user_id=tree["user_id"], world_id=tree["world_id"])
            ] = CommitTree.from_json(tree["tree_data"])
            get_logger_backend().debug(
                f"Commit tree loaded: {CommitIdentifier(user_id=tree['user_id'], world_id=tree['world_id'])}"
            )


async def save_commit_tree(user_id: str, world_id: str, commit_tree: CommitTree):
    """保存提交树到数据库"""
    try:
        user_id_uuid = UUID(user_id)
        world_id_uuid = UUID(world_id)
    except ValueError:
        raise ValueError("无效的ID格式")

    async with commit_tree_lock:
        await db.save_commit_tree(
            user_id=user_id_uuid,
            world_id=world_id_uuid,
            tree_data=commit_tree.to_json(),
        )


async def save_user_dict():
    """保存用户字典到数据库"""
    async with user_lock:
        users = [
            {
                "user_id": UUID(k),
                "username": v.username,
                "password_hash": v.password_hash,
                "token": v.token,
            }
            for k, v in user_dict.items()
        ]
        await db.save_users(users)


async def load_user_dict():
    """从数据库加载用户字典"""
    async with user_lock:
        users = await db.get_all_users()
        _user_dict = {
            str(user["user_id"]): User(
                user_id=str(user["user_id"]),
                username=user["username"],
                password_hash=user["password_hash"],
                token=user["token"],
            )
            for user in users
        }
        user_dict.update(_user_dict)
        get_logger_backend().debug(f"User dict loaded: {_user_dict}")


async def save_graph(user_id: str, world_id: str, commit_id: str, graph: Graph):
    """保存图到数据库"""
    try:
        user_id_uuid = UUID(user_id)
        world_id_uuid = UUID(world_id)
        commit_id_uuid = UUID(commit_id)
    except ValueError:
        raise ValueError("无效的ID格式")

    async with world_lock:
        json_data = await graph.to_json(
            user_id=user_id, world_id=world_id, commit_id=commit_id
        )
        await db.create_world_commit(
            commit_id=commit_id_uuid,
            world_id=world_id_uuid,
            parent_commit_id=None,  # TODO: 从commit tree获取
            graph_data=json_data,
            topic=graph.commit_metadata.topic,
            event_summary=graph.commit_metadata.event_summary,
        )


async def load_graph():
    """从数据库加载图"""
    async def load_graph_from_commit(commit: dict[str, Any]):
        try:
            user_id = str(commit["user_id"])
            world_id = str(commit["world_id"])
            commit_id = str(commit["commit_id"])
            
            json_data = commit["graph_data"]
            if json_data["user_id"] != user_id:
                get_logger_backend().error(
                    f"User id mismatch: {json_data['user_id']} != {user_id}"
                )
            if json_data["world_id"] != world_id:
                get_logger_backend().error(
                    f"World id mismatch: {json_data['world_id']} != {world_id}"
                )
            if json_data["commit_id"] != commit_id:
                get_logger_backend().error(
                    f"Commit id mismatch: {json_data['commit_id']} != {commit_id}"
                )

            llm_config = GLOBAL_LLM_CONFIG.copy()
            if json_data["llm_config"].get("language_type", None) is not None:
                llm_config.language_type = LanguageType(
                    json_data["llm_config"]["language_type"]
                )
            json_data["llm_config"] = llm_config.to_json()

            G = await Graph.from_json(
                data=json_data,
                embeddings=GLOBAL_EMBEDDINGS,
                annotation_params=GLOBAL_ANNOTATION_PARAMS,
            )

            async with world_lock:
                world_dict[
                    WorldIdentifier(
                        user_id=user_id, world_id=world_id, commit_id=commit_id
                    )
                ] = G

            get_logger_backend().debug(
                f"World loaded: ({user_id}, {world_id}, {commit_id}) language type: {G.llm_config.language_type}"
            )

            # 检查场景状态并初始化
            context = G.org_tree.layer_manager.group_chat_context
            scene_status = await context.get_groupchat_status()
            if scene_status == GroupChatStatus.NOT_STARTED:
                # 检查是否是第一个提交
                commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
                async with commit_tree_lock:
                    previous_commit_id = commit_trees_dict[commit_identifier].root_id
                    is_first_scene = previous_commit_id == commit_id

                # 创建场景任务
                scene_task = await scene_task_manager.create_task(
                    user_id, world_id, commit_id
                )

                # 启动场景初始化
                background_task = asyncio.create_task(
                    background_scene_initialization(
                        G,
                        user_id,
                        world_id,
                        commit_id,
                        is_first_scene,
                        previous_commit_id,
                    )
                )
                scene_task.set_task(background_task)
                get_logger_backend().debug(
                    f"Started scene initialization for ({user_id}, {world_id}, {commit_id})"
                )

        except Exception as e:
            get_logger_backend().debug(traceback.format_exc())
            get_logger_backend().error(
                f"Error in loading graph: {e}, skip loading graph"
            )

    # 获取所有世界
    worlds = await db.get_all_worlds()
    for world in worlds:
        # 获取世界的所有提交
        commits = await db.get_world_commits(world["world_id"])
        tasks = [load_graph_from_commit(commit) for commit in commits]
        await asyncio.gather(*tasks)


ASYNC_SLEEP_TIME = 0.3


# 根路径重定向
@app.get("/")
async def root():
    return {"message": "Hello World"}


# OPTIONS方法支持
@app.options("/{full_path:path}")
async def options_route(full_path: str):
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept, Origin, X-Requested-With",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
        },
    )


# 注册(获得token)
@app.post("/auth/register", response_model=RegisterResponse)
async def register(request: Request):
    data = RegisterModel(**(await request.json()))
    if data.username in user_dict:
        raise HTTPException(status_code=400, detail="用户已存在")
    current_user_dict = data.model_dump()
    user_id = str(uuid4())
    current_user_dict.update({"user_id": user_id})
    token = create_access_token(current_user_dict)
    async with user_lock:
        user_dict[user_id] = User(
            token=token,
            user_id=user_id,
            username=data.username,
            password_hash=hash_password(data.password),  # 存储密码哈希
        )
    await save_user_dict()
    response = RegisterResponse(
        success=True,
        message="注册成功",
        token=token,
        id=user_id,
        user=current_user_dict,
    )
    return response


# 登录
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: Request):
    data = LoginModel(**(await request.json()))
    async with user_lock:
        user_name_2_id = {user.username: user.user_id for user in user_dict.values()}
    if data.username not in user_name_2_id:
        raise HTTPException(status_code=400, detail="用户不存在")
    async with user_lock:
        user = user_dict[user_name_2_id[data.username]]

    # 首先验证密码
    if user.password_hash is not None and not verify_password(
        data.password, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="密码错误")

    # 检查token是否过期
    try:
        user.decode_token()  # 尝试解码token
    except jwt.ExpiredSignatureError:
        # token过期，重新生成token
        user.update_token_with_password(data.password)
        async with user_lock:
            user_dict[user.user_id] = user
        await save_user_dict()
    except Exception as e:
        # 其他token错误，重新生成token
        user.update_token_with_password(data.password)
        async with user_lock:
            user_dict[user.user_id] = user
        await save_user_dict()

    return LoginResponse(
        success=True,
        message="登录成功",
        token=user.token,
        id=user.user_id,
        user={"id": user.user_id, "username": user.username},
    )


# 世界主页
@app.get("/user/{user_id}/world/{world_id}")
async def world_home(user_id: str, world_id: str):
    # 转换ID为UUID
    try:
        user_id_uuid = UUID(user_id)
        world_id_uuid = UUID(world_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的ID格式")

    # 获取世界信息
    world_data = await db.get_world(world_id_uuid)
    if not world_data:
        raise HTTPException(status_code=404, detail="世界不存在")

    # 获取该世界的所有commit
    commits = await db.get_world_commits(world_id_uuid)
    if not commits:
        raise HTTPException(status_code=404, detail="世界不存在commit")

    # 获取最新的commit
    latest_commit = commits[0]  # commits已按时间倒序排序
    latest_commit_parent = latest_commit["parent_commit_id"] if latest_commit["parent_commit_id"] else "root"

    # 从缓存中获取Graph对象
    commit_to_worlds = {}
    for commit in commits:
        world_identifier = WorldIdentifier(
            user_id=user_id,
            world_id=world_id,
            commit_id=str(commit["commit_id"])
        )
        if world_identifier in world_dict:
            commit_to_worlds[str(commit["commit_id"])] = world_dict[world_identifier]

    return {
        "user_id": user_id,
        "world_id": world_id,
        "commits": [
            {
                "commit_id": str(commit["commit_id"]),
                "topic": commit["topic"],
                "event_summary": commit["event_summary"],
                "parent_id": str(commit["parent_commit_id"]) if commit["parent_commit_id"] else "root",
            }
            for commit in commits
            if str(commit["commit_id"]) in commit_to_worlds
        ],
        "latest_commit": {
            "commit_id": str(latest_commit["commit_id"]),
            "topic": latest_commit["topic"],
            "event_summary": latest_commit["event_summary"],
            "parent_id": latest_commit_parent,
        },
    }


# 用户主页
@app.get("/user/{user_id}")
async def user_home(user_id: str):
    # 转换ID为UUID
    try:
        user_id_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的用户ID格式")

    # 获取用户信息
    user_data = await db.get_user_by_id(user_id_uuid)
    if not user_data:
        get_logger_backend().error(f"User {user_id} not found")
        raise HTTPException(status_code=404, detail="用户不存在")

    # 获取用户的所有世界
    user_worlds = await db.get_user_worlds(user_id_uuid)
    
    # 获取每个世界的最新commit
    worlds_with_commits = []
    for world in user_worlds:
        commits = await db.get_world_commits(world["world_id"])
        if commits:  # 如果有commit
            worlds_with_commits.append({
                "world_id": str(world["world_id"]),
                "commit_id": str(commits[0]["commit_id"])  # 使用最新的commit
            })
        else:
            worlds_with_commits.append({
                "world_id": str(world["world_id"]),
                "commit_id": None
            })

    return {
        "user_id": user_id,
        "username": user_data["username"],
        "worlds": worlds_with_commits,
    }


async def background_world_initialization(
    G: Graph,
    user_id: str,
    world_id: str,
    commit_id: str,
    character_image_downloader: CharacterImageDownloader,
):
    try:
        await G.init_world()
        task = await world_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_completed()

        # 标注角色sprite sheet 如果已经标注过则跳过
        await G.annotate_all_characters_sprite_sheet()

        # 下载角色图片 已下载的会跳过
        all_characters = await G.get_all_characters()
        temp_dir = "/tmp/character_images_temp"
        os.makedirs(temp_dir, exist_ok=True)

        for character in all_characters:
            # 创建临时文件路径
            temp_path = os.path.join(temp_dir, f"{character['id']}.png")
            temp_front_path = os.path.join(temp_dir, f"{character['id']}_front.png")
            
            # 下载图片到临时目录
            await character_image_downloader.download_character_image(
                params=character["sprite_sheet_annotation_string"],
                output_dir=temp_dir,
                output_filename=f"{character['id']}.png",
                front_output_filename=f"{character['id']}_front.png",
                regenerate=character.get("need_regenerate_sprite_sheet", False),
            )

            # 读取图片数据
            try:
                with open(temp_path, "rb") as f:
                    image_data = f.read()
                with open(temp_front_path, "rb") as f:
                    front_image_data = f.read()

                # 保存到数据库
                await save_character_image_to_db(
                    character_id=UUID(character["id"]),
                    image_data=image_data,
                    front_image_data=front_image_data,
                )

                # 清理临时文件
                os.remove(temp_path)
                os.remove(temp_front_path)
            except Exception as e:
                get_logger_backend().error(f"Failed to save character image: {e}")
                continue

        # 清理临时目录
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass

        for c in await G.character_map.get_all_characters():
            c.need_regenerate_sprite_sheet = False

        # 检查场景状态并自动开始场景
        context = G.org_tree.layer_manager.group_chat_context
        scene_status = await context.get_groupchat_status()
        if scene_status == GroupChatStatus.NOT_STARTED:
            # 根据commit tree判断当前commit是否是第一个commit
            commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
            async with commit_tree_lock:
                is_first_scene = (
                    commit_trees_dict[commit_identifier].root_id == commit_id
                )
                previous_commit_id = commit_trees_dict[commit_identifier].root_id
            # 创建场景任务
            scene_task = await scene_task_manager.create_task(
                user_id, world_id, commit_id
            )
            # 启动场景初始化
            background_task = asyncio.create_task(
                background_scene_initialization(
                    G, user_id, world_id, commit_id, is_first_scene, previous_commit_id
                )
            )
            scene_task.set_task(background_task)

    except Exception as e:
        get_logger_backend().error(f"World initialization failed: {e}")
        get_logger_backend().error(traceback.format_exc())
        task = await world_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_failed(str(e))


async def background_scene_initialization(
    G: Graph,
    user_id: str,
    world_id: str,
    commit_id: str,
    is_first_scene: bool,
    previous_commit_id: Optional[str] = None,
):
    try:
        get_logger_backend().debug(
            f"Starting scene initialization for ({user_id}, {world_id}, {commit_id})"
        )
        manager_fast_chat_llm_client = scene_task_manager.fast_chat_llm_client
        if manager_fast_chat_llm_client is None:
            fast_chat_llm_client = None
        else:
            fast_chat_llm_client = manager_fast_chat_llm_client.copy()
        current_scene = await start_scene_from_graph(
            G=G,
            character_image_output_path=os.path.join(
                CHARACTER_IMAGES_PATH, user_id, world_id, commit_id
            ),
            generated_character_image_output_path=(
                os.path.join(
                    CHARACTER_IMAGES_PATH, user_id, world_id, previous_commit_id
                )
                if previous_commit_id is not None
                else ""
            ),
            character_image_downloader=GLOBAL_CHARACTER_IMAGE_DOWNLOADER,
            annotation_params=GLOBAL_ANNOTATION_PARAMS,
            is_first_scene=is_first_scene,
            fast_chat_llm_client=fast_chat_llm_client,
        )
        get_logger_backend().debug(
            f"Scene initialization completed for ({user_id}, {world_id}, {commit_id})"
        )
        # 保存world
        await save_graph(
            user_id=user_id, world_id=world_id, commit_id=commit_id, graph=G
        )
        task = await scene_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_completed()
            get_logger_backend().debug(
                f"Scene task marked as completed for ({user_id}, {world_id}, {commit_id})"
            )
    except Exception as e:
        get_logger_backend().error(
            f"Scene initialization failed for ({user_id}, {world_id}, {commit_id}): {e}"
        )
        get_logger_backend().error(traceback.format_exc())
        task = await scene_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_failed(str(e))


# World
# seed prompt -> world
@app.post(
    "/world/seed_prompt_to_world",
)
async def seed_prompt_to_world(
    request: Request, user_id: str = Depends(get_current_user)
):
    data = SeedPromptToWorldModel(**(await request.json()))
    if data.user_id != user_id:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    
    try:
        # 转换ID为UUID
        user_id_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的用户ID格式")

    try:
        # 生成世界元数据
        llm_client = LLMClient()
        llm_config = GLOBAL_LLM_CONFIG.copy()
        if data.language_type is not None:
            llm_config.language_type = LanguageType(data.language_type)
        llm_client.set_llm_config(llm_config)
        get_logger_backend().debug(f"Seed prompt request: {data.model_dump()}")
        universe_metadata = await seed_prompt_to_universe_metadata(
            data.seed_prompt, llm_client, llm_config
        )
        get_logger_backend().debug(
            f"World metadata from seed prompt: {universe_metadata}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成世界失败: {e}")

    # 创建Graph对象
    G = Graph(
        protagonist_description=universe_metadata.protagonist_description,
        world_state=universe_metadata.world_state,
        strategy=universe_metadata.strategy,
        tone=(
            universe_metadata.tone if universe_metadata.tone is not None else "neutral"
        ),
        llm_config=llm_config,
        embeddings=GLOBAL_EMBEDDINGS,
        annotation_params=GLOBAL_ANNOTATION_PARAMS,
    )

    # 生成ID
    world_id = str(uuid4())
    world_id_uuid = UUID(world_id)
    commit_id = await G.generate_world_status_uuid()
    commit_id_uuid = UUID(commit_id)

    # 创建世界记录
    await db.create_world(
        world_id=world_id_uuid,
        user_id=user_id_uuid,
        title=universe_metadata.world_title,
        crisis=universe_metadata.world_crisis,
    )

    # 创建提交记录
    graph_data = await G.to_json(user_id=user_id, world_id=world_id, commit_id=commit_id)
    await db.create_world_commit(
        commit_id=commit_id_uuid,
        world_id=world_id_uuid,
        parent_commit_id=None,
        graph_data=graph_data,
        topic=G.commit_metadata.topic,
        event_summary=G.commit_metadata.event_summary,
    )

    # 更新提交树
    commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
    if commit_identifier not in commit_trees_dict:
        commit_trees_dict[commit_identifier] = CommitTree()
    commit_tree = commit_trees_dict[commit_identifier]
    await commit_tree.add_commit(
        world_id=world_id, user_id=user_id, commit_id=commit_id, graph=G, parent_id=None
    )
    await save_commit_tree(user_id, world_id, commit_tree)

    # 创建任务并存储图
    task = await world_task_manager.create_task(user_id, world_id, commit_id)
    world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ] = G

    # 设置初始权限
    permission_id = uuid4()
    await db.set_world_permission(
        permission_id=permission_id,
        world_id=world_id_uuid,
        commit_id=commit_id_uuid,
        owner_id=user_id_uuid,
        visibility=WorldVisibility.PRIVATE,
    )

    # 启动后台初始化
    background_task = asyncio.create_task(
        background_world_initialization(
            G, user_id, world_id, commit_id, GLOBAL_CHARACTER_IMAGE_DOWNLOADER
        )
    )
    task.set_task(background_task)

    # 返回最小响应
    return {
        "user_id": user_id,
        "world_id": world_id,
        "commit_id": commit_id,
    }


# 创建一个world
@app.post("/world/create_world")
async def create_world(request: Request, user_id: str = Depends(get_current_user)):
    data = CreateWorldModel(**(await request.json()))
    if data.user_id != user_id:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    
    try:
        # 转换ID为UUID
        user_id_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的用户ID格式")

    get_logger_backend().debug(f"Create world: {data}")
    
    # 配置LLM
    llm_config = GLOBAL_LLM_CONFIG.copy()
    if data.language_type is not None:
        llm_config.language_type = LanguageType(data.language_type)
    
    # 创建Graph对象
    G = Graph(
        protagonist_description=data.protagonist_description,
        world_state=data.world_state,
        strategy=data.strategy,
        tone=data.tone if data.tone is not None else "neutral",
        llm_config=llm_config,
        embeddings=GLOBAL_EMBEDDINGS,
        annotation_params=GLOBAL_ANNOTATION_PARAMS,
    )

    # 生成ID
    world_id = str(uuid4())
    world_id_uuid = UUID(world_id)
    commit_id = await G.generate_world_status_uuid()
    commit_id_uuid = UUID(commit_id)

    # 创建世界记录
    await db.create_world(
        world_id=world_id_uuid,
        user_id=user_id_uuid,
        title=data.world_state,  # 使用world_state作为标题
        crisis=data.protagonist_description,  # 使用protagonist_description作为crisis
    )

    # 创建提交记录
    graph_data = await G.to_json(user_id=user_id, world_id=world_id, commit_id=commit_id)
    await db.create_world_commit(
        commit_id=commit_id_uuid,
        world_id=world_id_uuid,
        parent_commit_id=None,
        graph_data=graph_data,
        topic=G.commit_metadata.topic,
        event_summary=G.commit_metadata.event_summary,
    )

    # 更新提交树
    commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
    if commit_identifier not in commit_trees_dict:
        commit_trees_dict[commit_identifier] = CommitTree()
    commit_tree = commit_trees_dict[commit_identifier]
    await commit_tree.add_commit(
        world_id=world_id, user_id=user_id, commit_id=commit_id, graph=G, parent_id=None
    )
    await save_commit_tree(user_id, world_id, commit_tree)

    # 创建任务并存储图
    task = await world_task_manager.create_task(user_id, world_id, commit_id)
    world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ] = G

    # 设置初始权限
    permission_id = uuid4()
    await db.set_world_permission(
        permission_id=permission_id,
        world_id=world_id_uuid,
        commit_id=commit_id_uuid,
        owner_id=user_id_uuid,
        visibility=WorldVisibility.PRIVATE,
    )

    # 启动后台初始化
    background_task = asyncio.create_task(
        background_world_initialization(
            G, user_id, world_id, commit_id, GLOBAL_CHARACTER_IMAGE_DOWNLOADER
        )
    )
    task.set_task(background_task)

    # 返回最小响应
    return {
        "user_id": user_id,
        "world_id": world_id,
        "commit_id": commit_id,
    }


# 获取所有世界的identifier
@app.post("/world/get_all_worlds", response_model=list[WorldIdentifier])
async def get_all_worlds(request: Request, user_id: str = Depends(get_current_user)):
    data = GetAllWorldsModel(**(await request.json()))
    if data.user_id != user_id:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    return [
        WorldIdentifier(user_id=w.user_id, world_id=w.world_id, commit_id=w.commit_id)
        for w in world_dict
    ]


# Scene
# get events at current scene
@app.post("/{user_id}/{world_id}/{commit_id}/scene/events")
async def get_events(
    user_id: str,
    world_id: str,
    commit_id: str,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    world_identifier = WorldIdentifier(
        user_id=user_id, world_id=world_id, commit_id=commit_id
    )
    if world_identifier not in world_dict:
        # check if world id exists
        world_identifiers = [
            _id for _id in world_dict.keys() if _id.world_id == world_id
        ]
        if not world_identifiers:
            raise HTTPException(status_code=404, detail="World not found")
        else:
            raise HTTPException(status_code=404, detail="Commit not found")
    G = world_dict[world_identifier]
    context = G.org_tree.layer_manager.group_chat_context
    current_layer = await context.get_current_layer()
    # while current_layer is None:
    #     await asyncio.sleep(ASYNC_SLEEP_TIME)
    #     current_layer = await context.get_current_layer()
    if current_layer is None:
        raise HTTPException(status_code=400, detail="场景未开始")
    # status
    status = await context.get_groupchat_status()
    if status == GroupChatStatus.TERMINATED:
        raise HTTPException(status_code=400, detail="场景已结束")
    elif status == GroupChatStatus.WAITING_FOR_PLAYER_ACTION:
        pass
    elif status == GroupChatStatus.EXECUTING_PLAYER_ACTION:
        pass
    elif status == GroupChatStatus.NOT_STARTED:
        get_logger_backend().debug("Scene not started, start scene")
        # Check if scene is already being initialized
        scene_task = await scene_task_manager.get_task(user_id, world_id, commit_id)
        if scene_task and scene_task.is_in_progress():
            return {
                "user_id": user_id,
                "world_id": world_id,
                "commit_id": commit_id,
                "status": "initializing_scene",
            }
        elif scene_task and scene_task.is_failed():
            get_logger_backend().error(
                f"Scene initialization failed: {scene_task.error}"
            )
        # Start scene initialization
        task = await scene_task_manager.create_task(user_id, world_id, commit_id)
        # 根据commit tree判断当前commit是否是第一个commit
        commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
        async with commit_tree_lock:
            previous_commit_id = commit_trees_dict[commit_identifier].root_id
            is_first_scene = previous_commit_id == commit_id
        background_task = asyncio.create_task(
            background_scene_initialization(
                G, user_id, world_id, commit_id, is_first_scene, previous_commit_id
            )
        )
        task.set_task(background_task)
        return {
            "user_id": user_id,
            "world_id": world_id,
            "commit_id": commit_id,
            "status": "initializing_scene",
        }
    elif status == GroupChatStatus.STARTED:
        pass
    else:
        raise HTTPException(status_code=400, detail="场景状态未知")
    # wait for dialogues
    dialogues = await context.get_dialogues()
    if dialogues is None:
        raise HTTPException(status_code=400, detail="对话未生成")
    # wait for options
    options = await context.get_options()
    if options is None:
        raise HTTPException(status_code=400, detail="选项未生成")
    # classify dialogues and options into events
    classified_events = [
        e for m in dialogues + [options] for e in message_to_event_model(m)
    ]
    # participants
    participants = await context.get_scene_participants()
    if participants is None:
        raise HTTPException(status_code=400, detail="角色未生成")
    # get missions
    missions = await context.get_scene_missions()
    assert missions is not None, "Missions not found"
    _mission_jsons = [m.to_json() for m in missions]
    processed_missions = [
        MissionModel(
            id=m["id"],
            name=m["name"],
            description=m["description"],
            status=m["status"],
            mission_type=m["mission_type"],
        )
        for m in _mission_jsons
    ]
    # scene meta
    scene_meta = current_layer.layer_manager.scene_metadata
    response = SceneModel(
        heading=scene_meta.heading,
        location=f"{scene_meta._location}",
        characterIds=[c.id for c in participants],
        eventList=classified_events,  # type: ignore
        missionList=processed_missions,
    )
    return response


@app.post(
    "/{user_id}/{world_id}/{commit_id}/scene/is-event-generated", response_model=bool
)
async def is_event_generated(
    user_id: str,
    world_id: str,
    commit_id: str,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    world_identifier = WorldIdentifier(
        user_id=user_id, world_id=world_id, commit_id=commit_id
    )
    if world_identifier not in world_dict:
        # check if world id exists
        world_identifiers = [
            _id for _id in world_dict.keys() if _id.world_id == world_id
        ]
        if not world_identifiers:
            raise HTTPException(status_code=404, detail="World not found")
        else:
            raise HTTPException(status_code=404, detail="Commit not found")
    G = world_dict[world_identifier]
    context = G.org_tree.layer_manager.group_chat_context
    current_layer = await context.get_current_layer()
    if current_layer is None:
        return False
    # status
    status = await context.get_groupchat_status()
    if status == GroupChatStatus.TERMINATED:
        raise HTTPException(status_code=400, detail="场景已结束")
    elif status == GroupChatStatus.WAITING_FOR_PLAYER_ACTION:
        pass
    elif status == GroupChatStatus.EXECUTING_PLAYER_ACTION:
        pass
    elif status == GroupChatStatus.NOT_STARTED:
        get_logger_backend().debug("Scene not started, start scene")
        # Check if scene is already being initialized
        scene_task = await scene_task_manager.get_task(user_id, world_id, commit_id)
        if scene_task and scene_task.is_in_progress():
            return False
        elif scene_task and scene_task.is_failed():
            get_logger_backend().error(
                f"Scene initialization failed: {scene_task.error}"
            )
        # Start scene initialization
        task = await scene_task_manager.create_task(user_id, world_id, commit_id)
        # 根据commit tree判断当前commit是否是第一个commit
        commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
        async with commit_tree_lock:
            previous_commit_id = commit_trees_dict[commit_identifier].root_id
            is_first_scene = previous_commit_id == commit_id
        background_task = asyncio.create_task(
            background_scene_initialization(
                G, user_id, world_id, commit_id, is_first_scene, previous_commit_id
            )
        )
        task.set_task(background_task)
        return False
    elif status == "started":
        pass
    else:
        raise HTTPException(status_code=400, detail="场景状态未知")
    # chat round
    chat_round = await context.get_scene_chat_round()
    last_updated_chat_round = await context.get_last_updated_chat_round()
    if chat_round == last_updated_chat_round:
        return True
    else:
        return False


# 选择选项
@app.post(
    "/{user_id}/{world_id}/{commit_id}/scene/select-option",
)
async def select_option(
    user_id: str,
    world_id: str,
    commit_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    data = SelectOptionModel(**(await request.json()))
    G = world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ]
    context = G.org_tree.layer_manager.group_chat_context
    # set input option index
    await context.set_selected_option_index(data.option_index)
    return {"status": "success"}


# 执行action
@app.post(
    "/{user_id}/{world_id}/{commit_id}/scene/input-action",
)
async def input_action(
    user_id: str,
    world_id: str,
    commit_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    data = InputActionModel(**(await request.json()))
    G = world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ]
    context = G.org_tree.layer_manager.group_chat_context
    await context.set_input_string(data.action)
    return {"status": "success"}


# 获取群聊是否已结束
@app.post(
    "/{user_id}/{world_id}/{commit_id}/scene/is_finished",
)
async def is_scene_finished(
    user_id: str,
    world_id: str,
    commit_id: str,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    G = world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ]
    context = G.org_tree.layer_manager.group_chat_context
    scene_status = await context.get_groupchat_status()
    if scene_status == GroupChatStatus.TERMINATED:
        # Check if commit creation is already in progress
        commit_task = await commit_task_manager.get_task(user_id, world_id, commit_id)
        if commit_task and commit_task.is_in_progress():
            return {"is_finished": True, "status": "creating_new_commit"}
        elif commit_task and commit_task.is_failed():
            raise HTTPException(
                status_code=500,
                detail=f"Commit creation failed: {commit_task.error}",
            )
        elif commit_task and commit_task.is_completed():
            new_commit_id = commit_task.commit_id
            return {"is_finished": True, "commit_id": new_commit_id}

        # Start commit creation
        new_commit_id = await G.generate_world_status_uuid()
        task = await commit_task_manager.create_task(user_id, world_id, commit_id)
        background_task = asyncio.create_task(
            background_commit_creation(G, user_id, world_id, commit_id, new_commit_id)
        )
        task.set_task(background_task)
        return {"is_finished": True, "status": "creating_new_commit"}
    else:
        return {"is_finished": False}


# 获取指定id的角色列表
@app.get(
    "/{user_id}/{world_id}/{commit_id}/world/get_characters",
    response_model=list[CharacterModel],
)
async def get_characters(
    user_id: str,
    world_id: str,
    commit_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    data = GetCharactersModel(**(await request.json()))
    G = world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ]
    characters = await G.get_all_characters()
    set_ids = set(data.ids)
    characters_list = [
        character_info_to_model(c) for c in characters if c["id"] in set_ids
    ]
    return characters_list


# 获取全部角色列表
@app.get(
    "/{user_id}/{world_id}/{commit_id}/world/get_all_characters",
    response_model=list[CharacterModel],
)
async def get_all_characters(
    user_id: str,
    world_id: str,
    commit_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    # data = GetAllCharactersModel(**(await request.json()))
    G = world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ]
    characters = await G.get_all_characters()
    return [character_info_to_model(c) for c in characters]


# 获取玩家角色
@app.get(
    "/{user_id}/{world_id}/{commit_id}/world/get_player_character",
    response_model=CharacterModel,
)
async def get_player_character(
    user_id: str,
    world_id: str,
    commit_id: str,
    current_user: str = Depends(get_current_user),
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="user_id不匹配")
    G = world_dict[
        WorldIdentifier(user_id=user_id, world_id=world_id, commit_id=commit_id)
    ]
    player_character = await G.get_player_character()
    return character_info_to_model(player_character)


@app.get("/user/{user_id}/world/{world_id}/commit/{commit_id}")
async def world_commit(user_id: str, world_id: str, commit_id: str):
    world_identifier = WorldIdentifier(
        user_id=user_id, world_id=world_id, commit_id=commit_id
    )
    G = world_dict.get(world_identifier)
    if G is None:
        raise HTTPException(status_code=404, detail="World not found")

    # Check if world is still being initialized
    world_task = await world_task_manager.get_task(user_id, world_id, commit_id)
    if world_task and world_task.is_in_progress():
        return {
            "user_id": user_id,
            "world_id": world_id,
            "commit_id": commit_id,
            "status": "initializing_world",
        }
    elif world_task and world_task.is_failed():
        raise HTTPException(
            status_code=500, detail=f"World initialization failed: {world_task.error}"
        )

    # World is initialized, check scene status
    context = G.org_tree.layer_manager.group_chat_context
    scene_status = await context.get_groupchat_status()
    if scene_status == GroupChatStatus.NOT_STARTED:
        # Check if scene is already being initialized
        scene_task = await scene_task_manager.get_task(user_id, world_id, commit_id)
        if scene_task and scene_task.is_in_progress():
            # 检查任务是否真的在运行
            if (
                not hasattr(scene_task, "_task")
                or scene_task._task is None
                or scene_task._task.done()
            ):
                # 如果任务不存在或已完成但没有更新状态，重新启动初始化
                get_logger_backend().debug(
                    f"Reactivating scene initialization for ({user_id}, {world_id}, {commit_id})"
                )
                # 根据commit tree判断当前commit是否是第一个commit
                commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
                async with commit_tree_lock:
                    previous_commit_id = commit_trees_dict[commit_identifier].root_id
                    is_first_scene = previous_commit_id == commit_id
                # 重新启动场景初始化
                background_task = asyncio.create_task(
                    background_scene_initialization(
                        G,
                        user_id,
                        world_id,
                        commit_id,
                        is_first_scene,
                        previous_commit_id,
                    )
                )
                scene_task.set_task(background_task)
            return {
                "user_id": user_id,
                "world_id": world_id,
                "commit_id": commit_id,
                "status": "initializing_scene",
            }
        elif scene_task and scene_task.is_failed():
            get_logger_backend().error(
                f"Scene initialization failed: {scene_task.error}"
            )
        # Start scene initialization
        task = await scene_task_manager.create_task(user_id, world_id, commit_id)
        # 根据commit tree判断当前commit是否是第一个commit
        commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
        async with commit_tree_lock:
            previous_commit_id = commit_trees_dict[commit_identifier].root_id
            is_first_scene = previous_commit_id == commit_id
        background_task = asyncio.create_task(
            background_scene_initialization(
                G, user_id, world_id, commit_id, is_first_scene, previous_commit_id
            )
        )
        task.set_task(background_task)
        return {
            "user_id": user_id,
            "world_id": world_id,
            "commit_id": commit_id,
            "status": "initializing_scene",
        }
    else:
        # wait for dialogues
        dialogues = await context.get_dialogues()
        if dialogues is None:
            return {
                "status": "waiting_for_dialogues",
            }
        # wait for options
        options = await context.get_options()
        if options is None:
            return {
                "status": "waiting_for_options",
            }
        classified_events = [
            e for m in dialogues + [options] for e in message_to_event_model(m)
        ]
        # get missions
        missions = await context.get_scene_missions()
        assert missions is not None, "Missions not found"
        _mission_jsons = [m.to_json() for m in missions]
        processed_missions = [
            MissionModel(
                id=m["id"],
                name=m["name"],
                description=m["description"],
                status=m["status"],
                mission_type=m["mission_type"],
            )
            for m in _mission_jsons
        ]
        # get participants
        participants = await context.get_scene_participants()
        assert participants is not None, "Participants not found"
        scene_meta = G.org_tree.layer_manager.scene_metadata
        current_scene = SceneModel(
            heading=scene_meta.heading,
            location=f"{scene_meta._location}",
            characterIds=[c.id for c in participants],
            eventList=classified_events,  # type: ignore
            missionList=processed_missions,
        )

    # response: all characters
    all_characters_list = await G.get_all_characters()
    # get_logger_backend().debug(f"All characters: {all_characters_list}")
    all_characters = [character_info_to_model(c) for c in all_characters_list]
    # get_logger_backend().debug(f"All characters models: {all_characters}")
    # world meta
    world_meta = G.universe_metadata
    response = WorldModel(
        id=world_id,
        commit_id=commit_id,
        title=world_meta.world_title,
        crisis=world_meta.world_crisis,
        allCharacters=all_characters,
        currentScene=current_scene,
        worldNews=[
            WorldNewsModel(
                id=n.id,
                title=n.title,
                content=n.content,
                date=n.date,
                impact=n.impact,  # type: ignore
                category=n.category,  # type: ignore
                relatedLocation=n.related_location,
            )
            for n in G.world_news
        ],
        worldCharacteristics=[
            WorldCharacteristicModel(
                name=n.name,
                description=n.description,
            )
            for n in world_meta.world_characteristics
        ],
        forkFrom=[
            ForkRelationModel(
                user_id=f.user_id,
                world_id=f.world_id,
                commit_id=f.commit_id,
                timestamp=f.timestamp,
            )
            for f in G.fork_from
        ],
        forkTo=[
            ForkRelationModel(
                user_id=f.user_id,
                world_id=f.world_id,
                commit_id=f.commit_id,
                timestamp=f.timestamp,
            )
            for f in G.fork_to
        ],
    )
    # get_logger_backend().debug(f"World model: {response}")
    return response


@app.post("/world/delete_world_commit")
async def delete_world_commit(
    request: Request,
    current_user: str = Depends(get_current_user),
):
    data = DeleteWorldCommitModel(**(await request.json()))
    if current_user != data.user_id:
        raise HTTPException(status_code=403, detail="user_id不匹配 无权限删除")

    try:
        # 转换ID为UUID
        user_id_uuid = UUID(data.user_id)
        world_id_uuid = UUID(data.world_id)
        commit_id_uuid = UUID(data.commit_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的ID格式")

    # 检查提交是否存在
    commit = await db.get_world_commit(commit_id_uuid)
    if not commit:
        raise HTTPException(status_code=404, detail="World commit not found")

    # 删除提交记录
    await db.delete_world_commit(commit_id_uuid)

    # 从内存中删除
    world_identifier = WorldIdentifier(
        user_id=data.user_id, world_id=data.world_id, commit_id=data.commit_id
    )
    async with world_lock:
        if world_identifier in world_dict:
            world_dict.pop(world_identifier)

    # 更新提交树
    commit_identifier = CommitIdentifier(user_id=data.user_id, world_id=data.world_id)
    if commit_identifier in commit_trees_dict:
        async with commit_tree_lock:
            commit_trees_dict[commit_identifier].delete_commit(data.commit_id)

    return {"message": "World commit deleted successfully"}


async def background_commit_creation(
    G: Graph, user_id: str, world_id: str, commit_id: str, new_commit_id: str
):
    try:
        # save graph (for commit metadata)
        await save_graph(user_id, world_id, commit_id, G)
        # fork options
        await G.generate_fork_options()
        # create new world identifier
        new_world_identifier = WorldIdentifier(
            user_id=user_id, world_id=world_id, commit_id=new_commit_id
        )
        new_graph = await Graph.from_json(
            await G.to_json(user_id, world_id, new_commit_id),
            embeddings=GLOBAL_EMBEDDINGS,
            annotation_params=GLOBAL_ANNOTATION_PARAMS,
        )
        layer_manager = new_graph.org_tree.layer_manager
        if layer_manager.current_scene_topic is not None:
            layer_manager.previous_scene_topics.append(
                f"{layer_manager.current_scene_topic}"
            )
        layer_manager.current_scene_topic = None
        async with world_lock:
            world_dict[new_world_identifier] = new_graph
        await save_graph(user_id, world_id, new_commit_id, new_graph)
        # create new commit tree
        commit_identifier = CommitIdentifier(user_id=user_id, world_id=world_id)
        if commit_identifier not in commit_trees_dict:
            commit_trees_dict[commit_identifier] = CommitTree()
        commit_tree = commit_trees_dict[commit_identifier]
        await commit_tree.add_commit(
            world_id=world_id,
            user_id=user_id,
            commit_id=new_commit_id,
            graph=new_graph,
            parent_id=commit_id,
        )
        await save_commit_tree(user_id, world_id, commit_tree)
        # 从数据库加载旧的世界到world_dict
        try:
            commit_id_uuid = UUID(commit_id)
        except ValueError:
            raise ValueError("无效的提交ID格式")

        old_commit = await db.get_world_commit(commit_id_uuid)
        if not old_commit:
            raise ValueError(f"找不到提交记录: {commit_id}")

        old_world_identifier = WorldIdentifier(
            user_id=user_id, world_id=world_id, commit_id=commit_id
        )
        old_graph = await Graph.from_json(
            old_commit["graph_data"],
            embeddings=GLOBAL_EMBEDDINGS,
            annotation_params=GLOBAL_ANNOTATION_PARAMS,
        )
        world_dict[old_world_identifier] = old_graph
        # clear group chat context
        await G.org_tree.layer_manager.group_chat_context.clear_all()

        # Add permission for new commit
        old_permission = await db.get_world_permission(UUID(commit_id))
        permission_id = uuid4()
        if old_permission:
            await db.set_world_permission(
                permission_id=permission_id,
                world_id=UUID(world_id),
                commit_id=UUID(new_commit_id),
                owner_id=UUID(user_id),
                visibility=WorldVisibility(old_permission["visibility"]),
                shared_with=old_permission["shared_with"],
            )
        else:
            await db.set_world_permission(
                permission_id=permission_id,
                world_id=UUID(world_id),
                commit_id=UUID(new_commit_id),
                owner_id=UUID(user_id),
                visibility=WorldVisibility.PRIVATE,
            )

        task = await commit_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_completed()
    except Exception as e:
        get_logger_backend().error(f"Commit creation failed: {e}")
        task = await commit_task_manager.get_task(user_id, world_id, commit_id)
        if task:
            task.set_failed(str(e))


@app.post("/world/public_world")
async def public_world(request: Request, current_user: str = Depends(get_current_user)):
    data = PublicWorldModel(**(await request.json()))
    if current_user != data.user_id:
        raise HTTPException(status_code=403, detail="user_id不匹配，无权限公开")
    world_identifier = WorldIdentifier(
        user_id=data.user_id, world_id=data.world_id, commit_id=data.commit_id
    )
    if world_identifier not in world_dict:
        raise HTTPException(status_code=404, detail="World commit not found")

    # Create new world ID for public world
    new_world_id = str(uuid4())

    # Set world visibility to public
    permission_id = uuid4()
    await db.set_world_permission(
        permission_id=permission_id,
        world_id=UUID(data.world_id),
        commit_id=UUID(data.commit_id),
        owner_id=UUID(data.user_id),
        visibility=WorldVisibility.PUBLIC,
    )
    # Create fork task
    task = await fork_task_manager.create_task(
        user_id=data.user_id,  # Use original user as owner
        world_id=data.world_id,
        commit_id=data.commit_id,
        new_world_id=new_world_id,
    )
    async with world_lock:
        G = world_dict[world_identifier]
    # Start background fork process
    background_task = asyncio.create_task(
        background_fork_world(
            world_dict=world_dict,
            world_lock=world_lock,
            commit_tree_lock=commit_tree_lock,
            commit_trees_dict=commit_trees_dict,
            source_graph=world_dict[world_identifier],
            user_id=data.user_id,
            world_id=data.world_id,
            commit_id=data.commit_id,
            new_user_id=data.user_id,  # Use original user as owner
            new_world_id=new_world_id,
            llm_client=G.llm_client,
            llm_config=G.llm_config,
            character_image_downloader=GLOBAL_CHARACTER_IMAGE_DOWNLOADER,
            character_images_path=CHARACTER_IMAGES_PATH,
            embeddings=GLOBAL_EMBEDDINGS,
            annotation_params=GLOBAL_ANNOTATION_PARAMS,
            fork_seed_prompt=None,
            mode="full",
        )
    )
    task.set_task(background_task)

    return {
        "user_id": data.user_id,
        "world_id": new_world_id,
        "status": "publishing_world",
    }


@app.post("/world/fork")
async def fork_world(request: Request, current_user: str = Depends(get_current_user)):
    data = ForkWorldModel(**(await request.json()))
    world_identifier = WorldIdentifier(
        user_id=data.user_id, world_id=data.world_id, commit_id=data.commit_id
    )
    if world_identifier not in world_dict:
        get_logger_backend().error(
            f"World commit not found: {world_identifier}, all identifiers: {world_dict.keys()}"
        )
        raise HTTPException(status_code=404, detail="World commit not found")

    # Check if user has access to the world
    try:
        commit_id_uuid = UUID(data.commit_id)
        current_user_uuid = UUID(current_user)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的ID格式")

    if not await db.can_access_world(commit_id_uuid, current_user_uuid):
        raise HTTPException(status_code=403, detail="无权限访问该世界")

    # Create new world ID
    new_world_id = str(uuid4())

    # Create fork task
    task = await fork_task_manager.create_task(
        user_id=current_user,
        world_id=data.world_id,
        commit_id=data.commit_id,
        new_world_id=new_world_id,
    )
    async with world_lock:
        G = world_dict[world_identifier]

    # Start background fork process
    background_task = asyncio.create_task(
        background_fork_world(
            world_dict=world_dict,
            world_lock=world_lock,
            commit_tree_lock=commit_tree_lock,
            commit_trees_dict=commit_trees_dict,
            source_graph=world_dict[world_identifier],
            user_id=data.user_id,
            world_id=data.world_id,
            commit_id=data.commit_id,
            new_user_id=current_user,
            new_world_id=new_world_id,
            llm_client=G.llm_client,
            llm_config=G.llm_config,
            character_image_downloader=GLOBAL_CHARACTER_IMAGE_DOWNLOADER,
            character_images_path=CHARACTER_IMAGES_PATH,
            embeddings=GLOBAL_EMBEDDINGS,
            annotation_params=GLOBAL_ANNOTATION_PARAMS,
            fork_seed_prompt=data.fork_seed_prompt,
            mode=data.mode,
        )
    )
    task.set_task(background_task)

    return {
        "user_id": current_user,
        "world_id": new_world_id,
        "status": "forking_world",
    }


@app.get("/world/public_worlds")
async def get_all_public_worlds(
    current_user: str = Depends(get_current_user), response_model=list[WorldIdentifier]
):
    try:
        current_user_uuid = UUID(current_user)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的用户ID格式")

    # 获取所有世界
    all_worlds = await db.get_all_worlds()
    
    # 检查每个世界的权限
    public_worlds = []
    for world in all_worlds:
        # 获取世界的最新commit
        commits = await db.get_world_commits(world["world_id"])
        if not commits:
            continue
            
        latest_commit = commits[0]  # commits已按时间倒序排序
        if await db.can_access_world(latest_commit["commit_id"], current_user_uuid):
            public_worlds.append(
                WorldIdentifier(
                    user_id=str(world["user_id"]),
                    world_id=str(world["world_id"]),
                    commit_id=str(latest_commit["commit_id"])
                )
            )

    return public_worlds


@app.get("/character/{user_id}/{world_id}/{commit_id}/{character_id}/portrait")
async def get_character_portrait(
    user_id: str,
    world_id: str,
    commit_id: str,
    character_id: str,
    current_user: str = Depends(get_current_user),
):
    # 转换ID为UUID
    try:
        commit_id_uuid = UUID(commit_id)
        character_id_uuid = UUID(character_id)
        current_user_uuid = UUID(current_user)
    except ValueError:
        raise HTTPException(status_code=400, detail="无效的ID格式")

    # 检查访问权限
    if not await db.can_access_world(commit_id_uuid, current_user_uuid):
        raise HTTPException(status_code=403, detail="无权限访问该世界")

    # 获取角色图片
    images = await db.get_character_images(character_id_uuid)
    if not images or "front_image_data" not in images:
        raise HTTPException(status_code=404, detail="Character portrait not found")
    
    # 创建临时文件并返回
    temp_dir = "/tmp/character_portraits"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = f"{temp_dir}/{character_id}_front.png"
    
    with open(temp_file, "wb") as f:
        f.write(images["front_image_data"])
    
    return FileResponse(
        temp_file,
        media_type="image/png",
        filename=f"{character_id}.png",
        background=BackgroundTask(cleanup_temp_file, temp_file)
    )

async def cleanup_temp_file(file_path: str):
    """清理临时文件的后台任务"""
    await asyncio.sleep(5)  # 等待文件被发送
    try:
        os.remove(file_path)
    except OSError:
        pass

async def save_character_image_to_db(
    character_id: UUID,
    image_data: bytes,
    front_image_data: bytes,
) -> None:
    """保存角色图片到数据库"""
    image_id = uuid4()
    await db.save_character_image(
        image_id=image_id,
        character_id=character_id,
        image_data=image_data,
        front_image_data=front_image_data,
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--dsn", type=str, default="")
    parser.add_argument("--fast_chat_api_key", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--fast_chat_model", type=str, default="")
    parser.add_argument("--provider", type=str, default="")
    parser.add_argument("--fast_chat_provider", type=str, default="")
    parser.add_argument("--proxies_port", type=int, default=-1)
    parser.add_argument("--ssl_keyfile", type=str, default="")
    parser.add_argument("--ssl_certfile", type=str, default="")
    parser.add_argument("--log_config", type=str, default="uvicorn_logconfig.json")
    return parser.parse_args()


# 启动服务
if __name__ == "__main__":
    import uvicorn

    args = get_args()
    if args.api_key != "":
        GLOBAL_LLM_CONFIG.api_key = args.api_key
    if args.model != "":
        GLOBAL_LLM_CONFIG.model = args.model
    if args.provider != "":
        GLOBAL_LLM_CONFIG.provider = LLMProvider(args.provider)
    if args.proxies_port != -1:
        get_logger_backend().info(f"Proxies: http://localhost:{args.proxies_port}")
        GLOBAL_LLM_CONFIG.proxies = {
            "http_proxy": f"http://localhost:{args.proxies_port}",
            "https_proxy": f"http://localhost:{args.proxies_port}",
        }
    if (
        args.fast_chat_api_key != ""
        and args.fast_chat_model != ""
        and args.fast_chat_provider != ""
    ):
        GLOBAL_FAST_CHAT_LLM_CONFIG.api_key = args.fast_chat_api_key
        GLOBAL_FAST_CHAT_LLM_CONFIG.model = args.fast_chat_model
        GLOBAL_FAST_CHAT_LLM_CONFIG.provider = LLMProvider(args.fast_chat_provider)
        fast_chat_llm_client = LLMClient(semaphore=100)
        fast_chat_llm_client.set_llm_config(GLOBAL_FAST_CHAT_LLM_CONFIG)
        scene_task_manager.fast_chat_llm_client = fast_chat_llm_client
        get_logger_backend().info(
            f"Fast chat LLM config: {GLOBAL_FAST_CHAT_LLM_CONFIG}"
        )
    get_logger_backend().debug(f"LLM config: {GLOBAL_LLM_CONFIG}")
    # connect to database
    asyncio.run(db.connect(args.dsn))
    asyncio.run(db.initialize_tables())
    asyncio.run(load_commit_trees())
    asyncio.run(load_user_dict())
    asyncio.run(load_graph())
    # support for https
    if args.ssl_keyfile != "" and args.ssl_certfile != "":
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            log_config=args.log_config,
        )
    else:
        # runs on http
        get_logger_backend().warning("Runs on http!")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_config=args.log_config,
        )
