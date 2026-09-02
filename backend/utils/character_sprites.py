import os
from typing import Any, Optional
from uuid import UUID

from AgentMatrix.src.graph import Graph
from AgentMatrix.src.spritesheet_generator import CharacterImageDownloader
from logger import get_logger

SPRITE_GENERATION_ENABLED = os.getenv("ENABLE_SPRITE_GENERATION", "").lower() in (
    "1",
    "true",
    "yes",
)


async def optional_prepare_character_sprites(
    graph: Graph,
    downloader: CharacterImageDownloader,
    output_dir: str,
    generated_image_dir: str = "",
    persist_to_db: Optional[Any] = None,
) -> None:
    """Annotate/download/save character sprites if possible. Never raise."""
    if not SPRITE_GENERATION_ENABLED:
        get_logger().debug("Sprite generation disabled; skipping character images")
        return

    try:
        await graph.annotate_all_characters_sprite_sheet()
    except Exception as e:
        get_logger().warning(f"Sprite annotation skipped: {e}")
        return

    try:
        characters = await graph.get_all_characters()
    except Exception as e:
        get_logger().warning(f"Could not list characters for sprite generation: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    for character in characters:
        character_id = character.get("id")
        if not character_id:
            continue
        try:
            produced = await downloader.download_character_image(
                params=character.get("sprite_sheet_annotation_string", ""),
                output_dir=output_dir,
                output_filename=f"{character_id}.png",
                front_output_filename=f"{character_id}_front.png",
                generated_image_path=(
                    os.path.join(generated_image_dir, f"{character_id}.png")
                    if generated_image_dir
                    else ""
                ),
                regenerate=character.get("need_regenerate_sprite_sheet", False),
            )
            if not produced:
                get_logger().debug(
                    f"No sprite files for character {character_id}; skipping save"
                )
                continue
            if persist_to_db is not None:
                image_path = os.path.join(output_dir, f"{character_id}.png")
                front_path = os.path.join(output_dir, f"{character_id}_front.png")
                if not (os.path.exists(image_path) and os.path.exists(front_path)):
                    continue
                with open(image_path, "rb") as handle:
                    image_data = handle.read()
                with open(front_path, "rb") as handle:
                    front_image_data = handle.read()
                await persist_to_db(
                    character_id=UUID(character_id),
                    image_data=image_data,
                    front_image_data=front_image_data,
                )
                try:
                    os.remove(image_path)
                    os.remove(front_path)
                except OSError:
                    pass
        except Exception as e:
            get_logger().warning(
                f"Sprite generation skipped for character {character_id}: {e}"
            )

    try:
        for character in await graph.character_map.get_all_characters():
            character.need_regenerate_sprite_sheet = False
    except Exception as e:
        get_logger().warning(f"Could not reset sprite regenerate flags: {e}")
