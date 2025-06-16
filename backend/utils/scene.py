import asyncio
from typing import Optional
import os

from AgentMatrix.model import MissionModel, SceneModel, message_to_event_model
from AgentMatrix.src.character import CharacterType
from AgentMatrix.src.graph import Graph, GroupChatStatus, HostLayer
from AgentMatrix.src.llm import LanguageType, LLMClient, LLMConfig, LLMProvider
from AgentMatrix.src.spritesheet_generator import (AnnotationParams,
                                                   CharacterImageDownloader)
from logger import get_logger

ASNYC_SLEEP_TIME = 0.3


async def start_scene_from_graph(
    G: Graph,
    character_image_downloader: CharacterImageDownloader,
    character_image_output_path: str,
    annotation_params: Optional[AnnotationParams] = None,
    is_first_scene: bool = False,
    fast_chat_llm_client: Optional[LLMClient] = None,
) -> SceneModel:
    # 标注角色sprite sheet 如果已经标注过则跳过
    await G.annotate_all_characters_sprite_sheet()

    # 下载角色图片 已下载的会跳过
    all_characters = await G.get_all_characters()
    download_tasks = [
        character_image_downloader.download_character_image(
            params=c["sprite_sheet_annotation_string"],
            output_dir=character_image_output_path,
            output_filename=f"{c['id']}.png",
            front_output_filename=f"{c['id']}_front.png",
        )
        for c in all_characters
    ]
    await asyncio.gather(*download_tasks)
    # start scene
    player_layer: HostLayer = None  # type: ignore
    universe_metadata = G.universe_metadata
    context = G.org_tree.layer_manager.group_chat_context
    for l in G.org_tree.layer_manager._layers_by_depth.values():
        l._active = True
        l_members = [
            m
            for n in l.nodes
            if n.organization is not None
            for m in await n.organization.get_members()
        ]
        if any(m.type == CharacterType.Player for m in l_members):
            player_layer = l
    assert player_layer is not None, "Player layer not found"
    asyncio.create_task(
        player_layer.group_chat(
            monitor=G.org_tree._monitor,
            world_state=universe_metadata.world_state,
            tone=universe_metadata.tone,
            commit_metadata=G.commit_metadata,
            fast_chat_llm_client=fast_chat_llm_client,
            is_first_scene=is_first_scene,
            terms=G.terms,
            character_image_downloader=character_image_downloader,
            character_image_output_path=character_image_output_path,
            annotation_params=annotation_params,
        )
    )
    await context.set_groupchat_status(GroupChatStatus.STARTED)
    # wait for dialogues
    dialogues = await context.get_dialogues()
    get_logger().debug(f"Start to get dialogues: {dialogues}")
    while dialogues is None:
        await asyncio.sleep(ASNYC_SLEEP_TIME)
        dialogues = await context.get_dialogues()
    get_logger().debug(f"Successfully got dialogues: {dialogues}")
    options = await context.get_options()
    get_logger().debug(f"Start to get options: {options}")
    while options is None:
        await asyncio.sleep(ASNYC_SLEEP_TIME)
        options = await context.get_options()
    get_logger().debug(f"Successfully got options: {options}")
    classified_events = [
        e for m in dialogues + [options] for e in message_to_event_model(m)
    ]
    # update participants status
    participants = await context.get_scene_participants()
    get_logger().debug(f"Start to get participants: {participants}")
    while participants is None:
        await asyncio.sleep(ASNYC_SLEEP_TIME)
        participants = await context.get_scene_participants()
    get_logger().debug(f"Successfully got participants: {participants}")
    # get missions
    missions = await context.get_scene_missions()
    get_logger().debug(f"Start to get missions: {missions}")
    assert missions is not None, "Missions not found"
    _mission_jsons = [m.to_json() for m in missions]
    get_logger().debug(f"Successfully got missions: {_mission_jsons}")
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
    scene_meta = player_layer.layer_manager.scene_metadata
    get_logger().debug(f"Start to get scene meta: {scene_meta}")
    return SceneModel(
        heading=scene_meta.heading,
        location=f"{scene_meta._location}",
        characterIds=[c.id for c in participants],
        eventList=classified_events,
        missionList=processed_missions,
    )
